// Global variables
let model;
let isTraining = false;

// Initialize application when window loads
window.onload = async function() {
    try {
        console.log('Initializing Matrix Factorization Recommender...');
        
        // Show immediate feedback
        updateResult('Starting application...', 'info');
        document.getElementById('predict-btn').disabled = true;
        
        // Load data with timeout protection
        const loadPromise = loadData();
        const timeoutPromise = new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Data loading timeout after 30 seconds')), 30000)
        );
        
        await Promise.race([loadPromise, timeoutPromise]);
        
        // Populate dropdowns immediately after data load
        populateUserDropdown();
        populateMovieDropdown();
        
        updateResult('Data loaded! Starting model training...', 'success');
        
        // Start model training
        await trainModel();
        
    } catch (error) {
        console.error('Initialization error:', error);
        updateResult(`Error: ${error.message}`, 'error');
    }
};

/**
 * Create Matrix Factorization model architecture
 */
function createModel(numUsers, numMovies, latentDim = 20) {
    console.log(`Creating model with ${numUsers} users, ${numMovies} movies, latent dimension: ${latentDim}`);
    
    // Input layers
    const userInput = tf.input({shape: [1], name: 'user_input'});
    const movieInput = tf.input({shape: [1], name: 'movie_input'});
    
    // Embedding layers
    const userEmbedding = tf.layers.embedding({
        inputDim: numUsers,
        outputDim: latentDim,
        inputLength: 1,
        name: 'user_embedding'
    }).apply(userInput);
    
    const movieEmbedding = tf.layers.embedding({
        inputDim: numMovies,
        outputDim: latentDim,
        inputLength: 1,
        name: 'movie_embedding'
    }).apply(movieInput);
    
    // Bias terms
    const userBias = tf.layers.embedding({
        inputDim: numUsers,
        outputDim: 1,
        inputLength: 1,
        name: 'user_bias'
    }).apply(userInput);
    
    const movieBias = tf.layers.embedding({
        inputDim: numMovies,
        outputDim: 1,
        inputLength: 1,
        name: 'movie_bias'
    }).apply(movieInput);
    
    // Dot product and prediction
    const dotProduct = tf.layers.dot({
        axes: -1,
        normalize: false,
        name: 'dot_product'
    }).apply([userEmbedding, movieEmbedding]);
    
    const prediction = tf.layers.add({
        name: 'prediction'
    }).apply([dotProduct, userBias, movieBias]);
    
    const output = tf.layers.flatten().apply(prediction);
    
    // Create and return the model
    const model = tf.model({
        inputs: [userInput, movieInput],
        outputs: output
    });
    
    console.log('Model architecture created successfully');
    return model;
}

/**
 * Train the matrix factorization model
 */
async function trainModel() {
    try {
        isTraining = true;
        document.getElementById('predict-btn').disabled = true;
        
        console.log('Starting model training...');
        updateResult('Training matrix factorization model...', 'info');
        
        // Create model
        model = createModel(numUsers, numMovies, 20);
        
        // Compile model
        model.compile({
            optimizer: tf.train.adam(0.01),
            loss: 'meanSquaredError',
            metrics: ['mse']
        });
        
        console.log('Model compiled successfully');
        
        // Prepare training data
        const userIds = ratings.map(r => r.userId);
        const movieIds = ratings.map(r => r.movieId);
        const ratingValues = ratings.map(r => r.rating);
        
        // Convert to tensors
        const userTensor = tf.tensor2d(userIds, [userIds.length, 1]);
        const movieTensor = tf.tensor2d(movieIds, [movieIds.length, 1]);
        const ratingTensor = tf.tensor2d(ratingValues, [ratingValues.length, 1]);
        
        console.log('Training data prepared:', {
            users: userIds.length,
            movies: movieIds.length,
            ratings: ratingValues.length
        });
        
        // Train the model
        await model.fit([userTensor, movieTensor], ratingTensor, {
            epochs: 5,
            batchSize: 128,
            validationSplit: 0.1,
            callbacks: {
                onEpochBegin: (epoch, logs) => {
                    updateTrainingProgress(epoch + 1, 5, 'Starting epoch...');
                },
                onEpochEnd: (epoch, logs) => {
                    const loss = logs.loss ? logs.loss.toFixed(4) : '--';
                    const valLoss = logs.val_loss ? logs.val_loss.toFixed(4) : '--';
                    updateTrainingProgress(epoch + 1, 5, `Loss: ${loss}, Val Loss: ${valLoss}`);
                    console.log(`Epoch ${epoch + 1} completed - Loss: ${loss}`);
                }
            }
        });
        
        // Clean up tensors
        userTensor.dispose();
        movieTensor.dispose();
        ratingTensor.dispose();
        
        // Update UI
        updateResult('Model trained successfully! Select a user and movie to predict ratings.', 'success');
        document.getElementById('predict-btn').disabled = false;
        isTraining = false;
        
        console.log('Model training completed');
        
    } catch (error) {
        console.error('Training error:', error);
        updateResult(`Training failed: ${error.message}`, 'error');
        isTraining = false;
    }
}

/**
 * Predict rating for selected user and movie
 */
async function predictRating() {
    if (!model || isTraining) {
        updateResult('Model is not ready yet. Please wait for training to complete.', 'warning');
        return;
    }
    
    try {
        const userId = parseInt(document.getElementById('user-select').value);
        const movieId = parseInt(document.getElementById('movie-select').value);
        
        if (isNaN(userId) || isNaN(movieId)) {
            updateResult('Please select both a user and a movie.', 'warning');
            return;
        }
        
        // Get movie title for display
        const movie = movies.find(m => m.id === movieId);
        const movieTitle = movie ? movie.title : `Movie ${movieId}`;
        
        updateResult(`Predicting rating for user ${userId} and "${movieTitle}"...`, 'info');
        
        // Create input tensors
        const userTensor = tf.tensor2d([[userId]], [1, 1]);
        const movieTensor = tf.tensor2d([[movieId]], [1, 1]);
        
        // Make prediction
        const prediction = model.predict([userTensor, movieTensor]);
        const rating = await prediction.data();
        const predictedRating = rating[0];
        
        // Clean up tensors
        userTensor.dispose();
        movieTensor.dispose();
        prediction.dispose();
        
        // Display result
        const displayRating = Math.min(Math.max(predictedRating, 0.5), 5).toFixed(2);
        const stars = '★'.repeat(Math.round(predictedRating)) + '☆'.repeat(5 - Math.round(predictedRating));
        
        updateResult(`
            <strong>Prediction Result:</strong><br>
            User ${userId} would rate "<strong>${movieTitle}</strong>"<br>
            <span class="rating-prediction">${displayRating}/5.0 ${stars}</span><br>
            <small>Matrix Factorization model prediction</small>
        `, 'success');
        
    } catch (error) {
        console.error('Prediction error:', error);
        updateResult(`Prediction failed: ${error.message}`, 'error');
    }
}

/**
 * UI Helper Functions
 */
function populateUserDropdown() {
    const select = document.getElementById('user-select');
    select.innerHTML = '';
    
    // Add users to dropdown
    const maxUsers = Math.min(100, numUsers);
    for (let i = 0; i < maxUsers; i++) {
        const option = document.createElement('option');
        option.value = i;
        option.textContent = `User ${i + 1}`;
        select.appendChild(option);
    }
    
    console.log(`Populated user dropdown with ${maxUsers} users`);
}

function populateMovieDropdown() {
    const select = document.getElementById('movie-select');
    select.innerHTML = '';
    
    // Add movies to dropdown
    movies.forEach(movie => {
        const option = document.createElement('option');
        option.value = movie.id;
        option.textContent = `${movie.title} (ID: ${movie.id})`;
        select.appendChild(option);
    });
    
    console.log(`Populated movie dropdown with ${movies.length} movies`);
}

function updateResult(message, type = 'info') {
    const resultElement = document.getElementById('result');
    
    const colors = {
        info: '#3498db',
        success: '#27ae60',
        error: '#e74c3c',
        warning: '#f39c12'
    };
    
    resultElement.style.borderLeftColor = colors[type] || colors.info;
    resultElement.innerHTML = `<p>${message}</p>`;
}

function updateTrainingProgress(epoch, totalEpochs, status) {
    const progress = (epoch / totalEpochs) * 100;
    document.getElementById('progress-fill').style.width = `${progress}%`;
    document.getElementById('epoch-info').textContent = `Epoch: ${epoch}/${totalEpochs}`;
    document.getElementById('loss-info').textContent = status;
}

// Export functions for global access
window.predictRating = predictRating;
