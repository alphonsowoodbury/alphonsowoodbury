```mermaid
%% Flashcard Creation Process
flowchart TD
    A[Select Challenge Row from Table] --> B[Create Flashcard Front]
    A --> C[Create Flashcard Back]
    B --> D[Front: Prompt + Challenge Name]
    C --> E[Back: DS/Algorithm Giveaways, Test-First Advice, Advanced Tips]
    D & E --> F[Active Recall and Review]
