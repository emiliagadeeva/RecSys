Role

-   You are an expert front‑end ML engineer building a browser‑based Two‑Tower retrieval demo with TensorFlow.js for the MovieLens 100K dataset (u.data, u.item, u.genre), suitable for static GitHub Pages hosting.

Context

-   Dataset: MovieLens 100K
    -   u.data format: user_id, item_id, rating, timestamp separated by tabs; 100k interactions; 943 users; 1,682 items.
    -   u.item format: item_id|title|release_date|video_release_date|IMDb_URL|genres; use item_id, title, and genres (19 binary genre features).
    -   u.user (optional): user_id|age|gender|occupation|zip_code for user features.

-   Goal: Build THREE in‑browser Two‑Tower models:
    1.  Deep Learning Two-Tower: user_id → embedding, item_id + genres → embedding
    2.  MLP Two-Tower: user_id → MLP with hidden layers, item_id + genres → MLP with hidden layers  
    3.  Baseline: Simple matrix factorization without deep learning

-   All models use dot product scoring and sampled softmax loss.

-   UX requirements:
    -   Buttons: "Load Data", "Train All Models", "Test", "Compare Models".
    -   Training shows separate loss charts for each model.
    -   Test action: show THREE side-by-side tables:
        - Left: User's historical top-10 rated movies
        - Middle: MLP model recommendations  
        - Right: Deep Learning model recommendations
    -   Add model comparison metrics (precision@k, recall@k).

-   New Technical Requirements:
    -   MLP Tower: At least one hidden layer with ReLU activation
    -   Genre Features: Use all 19 genre dimensions as item features
    -   User Features: Optionally use age, gender, occupation if u.user available
    -   Model Comparison: Quantitative comparison between all three approaches

Instructions

-   Return three files with complete code, each in a separate fenced code block.

a) index.html
-   Add model selection radio buttons (Deep Learning, MLP, Baseline)
-   Add "Compare Models" button and results area
-   Include three side-by-side tables for comparison
-   Add genre information display in item tables

b) app.js  
-   Load and parse genre information from u.item
-   Implement user feature processing if u.user available
-   Create three separate model instances
-   Add model comparison functionality with metrics
-   Update visualization to show genre clusters

c) two-tower.js
-   Implement THREE model architectures:
    1.  DeepLearningTwoTower: Standard deep embedding towers
    2.  MLPTwoTower: MLP with hidden layers in both towers
    3.  BaselineTwoTower: Simple embedding without deep layers
-   Add genre concatenation in item tower
-   Add optional user feature concatenation in user tower
-   Implement comparison metrics calculation

Format

-   Return three code blocks labeled exactly:
    - index.html
    - app.js  
    - two-tower.js
-   Ensure the code handles genre features and model comparison
-   UI must show three recommendation tables when comparing models
