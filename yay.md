# DynamoDB Service Implementation for Data Platform

## Service Architecture

Based on your serverless event-driven architecture, I've redesigned the DynamoDB service to align with your existing RunTable and FileTable schema and semantics.

```python
import boto3
import logging
from typing import Dict, List, Any, Optional, Literal, Union
from botocore.exceptions import ClientError
from datetime import datetime
from uuid import uuid4
from pydantic import BaseModel, Field, validator, root_validator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Core Models
class RunStatus(BaseModel):
    """Valid status values for a dataset acquisition run."""
    status: Literal["pending", "dispatched", "processing", "success", "failed", "cancelled"]

class FileStatus(BaseModel):
    """Valid status values for an individual file in processing."""
    status: Literal["pending", "fetched", "verified", "preprocessed", "failed"]

class RunTableItem(BaseModel):
    """Run-level tracking for dataset acquisition jobs."""
    runId: str
    datasetName: str
    businessDate: str
    sourceType: Literal["control-m", "s3", "manual", "internal"]
    sourceTrigger: Dict[str, Any]
    createdAt: str
    status: str
    resolvedFiles: int = 0
    processedFiles: int = 0
    environment: Literal["staging", "prod"]
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None
    
    @validator('status')
    def validate_status(cls, v):
        valid_statuses = ["pending", "dispatched", "processing", "success", "failed", "cancelled"]
        if v not in valid_statuses:
            raise ValueError(f"Invalid run status: {v}. Must be one of {valid_statuses}")
        return v
    
    @validator('createdAt')
    def validate_timestamp(cls, v):
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Invalid ISO timestamp format: {v}")
        return v
    
    class Config:
        extra = "forbid"
        
    def to_dynamo_item(self) -> Dict:
        """Convert model to DynamoDB item format."""
        return self.dict(exclude_none=True)

class FileTableItem(BaseModel):
    """File-level tracking for individual files in a dataset run."""
    fileId: str
    runId: str
    datasetName: str
    businessDate: str
    environment: Literal["staging", "prod"]
    location: str
    status: str
    lastUpdated: str
    fileSize: Optional[int] = None
    checksum: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None
    
    @validator('status')
    def validate_status(cls, v):
        valid_statuses = ["pending", "fetched", "verified", "preprocessed", "failed"]
        if v not in valid_statuses:
            raise ValueError(f"Invalid file status: {v}. Must be one of {valid_statuses}")
        return v
    
    @validator('lastUpdated')
    def validate_timestamp(cls, v):
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Invalid ISO timestamp format: {v}")
        return v
    
    @validator('location')
    def validate_location(cls, v):
        if not v.startswith(('s3://', 'https://', 'http://')):
            raise ValueError(f"Invalid location URI: {v}")
        return v
    
    class Config:
        extra = "forbid"
        
    def to_dynamo_item(self) -> Dict:
        """Convert model to DynamoDB item format."""
        return self.dict(exclude_none=True)
```

## Table Interfaces

```python
class RunTableService:
    """Interface for RunTable operations in the data pipeline."""
    
    def __init__(self, table):
        self.table = table
    
    def create_run(self, run_item: RunTableItem) -> Dict:
        """
        Create a new run record in RunTable.
        
        Args:
            run_item: Validated RunTableItem 
            
        Returns:
            Response from DynamoDB
        """
        try:
            response = self.table.put_item(
                Item=run_item.to_dynamo_item()
            )
            logger.info(f"Created run: {run_item.runId} for dataset {run_item.datasetName}")
            return response
        except ClientError as e:
            logger.error(f"Failed to create run: {e}")
            raise
    
    def update_run_status(self, 
                         run_id: str, 
                         status: str,
                         processed_files: Optional[int] = None,
                         error: Optional[Dict] = None) -> Dict:
        """
        Update run status and related attributes.
        
        Args:
            run_id: Unique run identifier
            status: New status value
            processed_files: Count of processed files
            error: Error details if failed
            
        Returns:
            Response from DynamoDB
        """
        # Validate the status
        RunStatus(status=status)
        
        update_expression = "SET #status = :status"
        expression_attribute_names = {'#status': 'status'}
        expression_attribute_values = {':status': status}
        
        if processed_files is not None:
            update_expression += ", processedFiles = :processed_files"
            expression_attribute_values[':processed_files'] = processed_files
            
        if error is not None:
            update_expression += ", #error = :error"
            expression_attribute_names['#error'] = 'error'
            expression_attribute_values[':error'] = error
        
        try:
            response = self.table.update_item(
                Key={'runId': run_id},
                UpdateExpression=update_expression,
                ExpressionAttributeNames=expression_attribute_names,
                ExpressionAttributeValues=expression_attribute_values,
                ReturnValues="UPDATED_NEW"
            )
            logger.info(f"Updated run status: {run_id} -> {status}")
            return response
        except ClientError as e:
            logger.error(f"Failed to update run status: {e}")
            raise
    
    def get_run(self, run_id: str, consistent_read: bool = False) -> Optional[RunTableItem]:
        """
        Retrieve run details.
        
        Args:
            run_id: Unique run identifier
            consistent_read: Whether to use strong consistency
            
        Returns:
            RunTableItem if found, None otherwise
        """
        try:
            response = self.table.get_item(
                Key={'runId': run_id},
                ConsistentRead=consistent_read
            )
            
            item = response.get('Item')
            if not item:
                return None
                
            return RunTableItem(**item)
        except ClientError as e:
            logger.error(f"Failed to retrieve run details: {e}")
            raise
    
    def query_runs_by_dataset_and_date(self, 
                                      dataset_name: str,
                                      business_date: str,
                                      limit: int = 10) -> List[RunTableItem]:
        """
        Query runs by dataset and business date using GSI.
        
        Args:
            dataset_name: Name of the dataset
            business_date: Business date in ISO format
            limit: Maximum items to return
            
        Returns:
            List of RunTableItem models
        """
        try:
            response = self.table.query(
                IndexName='datasetName-businessDate-index',
                KeyConditionExpression='datasetName = :ds AND businessDate = :bd',
                ExpressionAttributeValues={
                    ':ds': dataset_name,
                    ':bd': business_date
                },
                Limit=limit
            )
            
            return [RunTableItem(**item) for item in response.get('Items', [])]
        except ClientError as e:
            logger.error(f"Failed to query runs: {e}")
            raise
    
    def query_runs_by_status(self, 
                           status: str,
                           limit: int = 10) -> List[RunTableItem]:
        """
        Query runs by status using GSI.
        
        Args:
            status: Run status to filter by
            limit: Maximum items to return
            
        Returns:
            List of RunTableItem models
        """
        # Validate the status
        RunStatus(status=status)
        
        try:
            response = self.table.query(
                IndexName='status-index',
                KeyConditionExpression='#status = :status_val',
                ExpressionAttributeNames={'#status': 'status'},
                ExpressionAttributeValues={':status_val': status},
                Limit=limit
            )
            
            return [RunTableItem(**item) for item in response.get('Items', [])]
        except ClientError as e:
            logger.error(f"Failed to query runs by status: {e}")
            raise

class FileTableService:
    """Interface for FileTable operations in the data pipeline."""
    
    def __init__(self, table):
        self.table = table
    
    def create_file(self, file_item: FileTableItem) -> Dict:
        """
        Create a new file record in FileTable.
        
        Args:
            file_item: Validated FileTableItem
            
        Returns:
            Response from DynamoDB
        """
        try:
            response = self.table.put_item(
                Item=file_item.to_dynamo_item()
            )
            logger.info(f"Created file: {file_item.fileId} for run {file_item.runId}")
            return response
        except ClientError as e:
            logger.error(f"Failed to create file: {e}")
            raise
    
    def batch_create_files(self, file_items: List[FileTableItem]) -> Dict:
        """
        Create multiple file records in batch.
        
        Args:
            file_items: List of validated FileTableItems
            
        Returns:
            Dictionary with 'UnprocessedItems' if any
        """
        try:
            with self.table.batch_writer() as batch:
                for file_item in file_items:
                    batch.put_item(Item=file_item.to_dynamo_item())
            
            logger.info(f"Batch created {len(file_items)} files")
            return {'UnprocessedItems': {}}
        except ClientError as e:
            logger.error(f"Failed to batch create files: {e}")
            raise
    
    def update_file_status(self,
                          file_id: str,
                          status: str,
                          file_size: Optional[int] = None,
                          checksum: Optional[str] = None,
                          error: Optional[Dict] = None) -> Dict:
        """
        Update file status and metadata.
        
        Args:
            file_id: Unique file identifier
            status: New status value
            file_size: File size in bytes
            checksum: File checksum
            error: Error details if failed
            
        Returns:
            Response from DynamoDB
        """
        # Validate the status
        FileStatus(status=status)
        
        update_expression = "SET #status = :status, lastUpdated = :last_updated"
        expression_attribute_names = {'#status': 'status'}
        expression_attribute_values = {
            ':status': status,
            ':last_updated': datetime.now().isoformat()
        }
        
        if file_size is not None:
            update_expression += ", fileSize = :file_size"
            expression_attribute_values[':file_size'] = file_size
            
        if checksum is not None:
            update_expression += ", checksum = :checksum"
            expression_attribute_values[':checksum'] = checksum
            
        if error is not None:
            update_expression += ", #error = :error"
            expression_attribute_names['#error'] = 'error'
            expression_attribute_values[':error'] = error
        
        try:
            response = self.table.update_item(
                Key={'fileId': file_id},
                UpdateExpression=update_expression,
                ExpressionAttributeNames=expression_attribute_names,
                ExpressionAttributeValues=expression_attribute_values,
                ReturnValues="UPDATED_NEW"
            )
            logger.info(f"Updated file status: {file_id} -> {status}")
            return response
        except ClientError as e:
            logger.error(f"Failed to update file status: {e}")
            raise
    
    def get_file(self, file_id: str, consistent_read: bool = False) -> Optional[FileTableItem]:
        """
        Retrieve file details.
        
        Args:
            file_id: Unique file identifier
            consistent_read: Whether to use strong consistency
            
        Returns:
            FileTableItem if found, None otherwise
        """
        try:
            response = self.table.get_item(
                Key={'fileId': file_id},
                ConsistentRead=consistent_read
            )
            
            item = response.get('Item')
            if not item:
                return None
                
            return FileTableItem(**item)
        except ClientError as e:
            logger.error(f"Failed to retrieve file details: {e}")
            raise
    
    def query_files_by_run(self, run_id: str) -> List[FileTableItem]:
        """
        Query all files for a specific run.
        
        Args:
            run_id: Run identifier to filter by
            
        Returns:
            List of FileTableItem models
        """
        try:
            response = self.table.query(
                IndexName='runId-index',
                KeyConditionExpression='runId = :rid',
                ExpressionAttributeValues={':rid': run_id}
            )
            
            return [FileTableItem(**item) for item in response.get('Items', [])]
        except ClientError as e:
            logger.error(f"Failed to query files by run: {e}")
            raise
    
    def query_files_by_dataset_and_status(self,
                                         dataset_name: str,
                                         status: str) -> List[FileTableItem]:
        """
        Query files by dataset and status using GSI.
        
        Args:
            dataset_name: Name of the dataset
            status: File status to filter by
            
        Returns:
            List of FileTableItem models
        """
        # Validate the status
        FileStatus(status=status)
        
        try:
            response = self.table.query(
                IndexName='datasetName-status-index',
                KeyConditionExpression='datasetName = :ds AND #status = :status_val',
                ExpressionAttributeNames={'#status': 'status'},
                ExpressionAttributeValues={
                    ':ds': dataset_name,
                    ':status_val': status
                }
            )
            
            return [FileTableItem(**item) for item in response.get('Items', [])]
        except ClientError as e:
            logger.error(f"Failed to query files by dataset and status: {e}")
            raise

class DataPlatformDynamoDBService:
    """Main service for DynamoDB operations in the data platform."""
    
    def __init__(self, 
                run_table_name: str, 
                file_table_name: str,
                region_name: str = 'us-east-1'):
        """
        Initialize the DynamoDB service.
        
        Args:
            run_table_name: Name of RunTable
            file_table_name: Name of FileTable
            region_name: AWS region
        """
        self.dynamo_resource = boto3.resource('dynamodb', region_name=region_name)
        
        # Initialize tables
        self.run_table = self.dynamo_resource.Table(run_table_name)
        self.file_table = self.dynamo_resource.Table(file_table_name)
        
        # Initialize interfaces
        self.runs = RunTableService(self.run_table)
        self.files = FileTableService(self.file_table)
        
    def health_check(self) -> Dict[str, bool]:
        """
        Perform a health check on both tables.
        
        Returns:
            Dictionary with table health status
        """
        health = {}
        
        try:
            self.run_table.table_status
            health['run_table'] = True
        except Exception as e:
            logger.error(f"RunTable health check failed: {e}")
            health['run_table'] = False
            
        try:
            self.file_table.table_status
            health['file_table'] = True
        except Exception as e:
            logger.error(f"FileTable health check failed: {e}")
            health['file_table'] = False
            
        return health
    
    def generate_file_id(self, dataset_name: str, business_date: str, file_path: str) -> str:
        """
        Generate a deterministic file ID based on dataset, date and path.
        
        Args:
            dataset_name: Name of the dataset
            business_date: Business date
            file_path: Full path or URL to the file
            
        Returns:
            Hashed file ID string
        """
        import hashlib
        hash_input = f"{dataset_name}:{business_date}:{file_path}"
        return hashlib.md5(hash_input.encode('utf-8')).hexdigest()
```

## Lambda Integration

```python
import json
import os
from typing import List, Dict, Any
from datetime import datetime

# Assume imports from above code

def api_lambda_handler(event, context):
    """
    API Lambda handler for receiving dataset acquisition triggers.
    
    Args:
        event: Lambda event
        context: Lambda context
        
    Returns:
        Lambda response
    """
    # Initialize the DynamoDB service
    dynamo_service = DataPlatformDynamoDBService(
        run_table_name=os.environ.get('RUN_TABLE_NAME'),
        file_table_name=os.environ.get('FILE_TABLE_NAME')
    )
    
    # Extract metadata from the event
    dataset_name = event.get('datasetName')
    business_date = event.get('businessDate')
    source_type = event.get('sourceType', 'manual')
    environment = os.environ.get('ENVIRONMENT', 'staging')
    
    # Generate run ID (can be customized based on conventions)
    run_id = f"{dataset_name}-{business_date}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Create run record
    run_item = RunTableItem(
        runId=run_id,
        datasetName=dataset_name,
        businessDate=business_date,
        sourceType=source_type,
        sourceTrigger=event.get('sourceTrigger', {}),
        createdAt=datetime.now().isoformat(),
        status="pending",
        environment=environment,
        metadata=event.get('metadata', {})
    )
    
    dynamo_service.runs.create_run(run_item)
    
    # Resolve files based on dataset configuration
    # (This would typically involve fetching dataset config from another table or service)
    file_locations = resolve_file_locations(dataset_name, business_date, event)
    
    # Create file records
    file_items = []
    for location in file_locations:
        file_id = dynamo_service.generate_file_id(dataset_name, business_date, location)
        file_item = FileTableItem(
            fileId=file_id,
            runId=run_id,
            datasetName=dataset_name,
            businessDate=business_date,
            environment=environment,
            location=location,
            status="pending",
            lastUpdated=datetime.now().isoformat()
        )
        file_items.append(file_item)
    
    # Batch create file records
    dynamo_service.files.batch_create_files(file_items)
    
    # Update run with resolved file count
    dynamo_service.runs.update_run_status(
        run_id=run_id,
        status="dispatched",
        processed_files=0
    )
    
    # Dispatch to preprocessor (implementation depends on your architecture)
    dispatch_to_preprocessor(run_id)
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'runId': run_id,
            'datasetName': dataset_name,
            'businessDate': business_date,
            'fileCount': len(file_items)
        })
    }

def resolve_file_locations(dataset_name: str, business_date: str, event: Dict) -> List[str]:
    """
    Resolve file locations based on dataset configuration.
    This would be implemented according to your platform's logic.
    """
    # Placeholder implementation
    return []

def dispatch_to_preprocessor(run_id: str):
    """
    Dispatch the run to preprocessor Lambda or Step Function.
    This would be implemented according to your platform's architecture.
    """
    # Placeholder implementation
    pass
```

## Concise Overview

**DynamoDB Service Design for Data Platform**

1. **Model Layer**
   - `RunTableItem`: Models run-level acquisition tracking
   - `FileTableItem`: Models file-level processing tracking
   - Strong validation on statuses and timestamps

2. **RunTable Service**
   - Run lifecycle management (`pending` → `success`/`failed`)
   - Indexes for querying by dataset/date and status
   - File count tracking for progress monitoring

3. **FileTable Service**
   - File lifecycle tracking across processing stages
   - Batch creation for file sets in a run
   - Query interface for run-based and dataset-based analysis

4. **Core Platform Service**
   - Table interface composition for single API exposure
   - Deterministic fileId generation for deduplication
   - Health monitoring for operational reliability

5. **Lambda Integration Pattern**
   - Trigger reception and run initialization
   - File resolution and batch registration
   - Status transitions and downstream dispatching

This implementation aligns with your serverless event-driven architecture, providing idempotent operations, status tracking, and observability across your acquisition pipelines.
