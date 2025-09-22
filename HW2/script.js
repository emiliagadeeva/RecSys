// Initialize the application when the window loads
window.onload = async function() {
    try {
        await loadData();
        populateMoviesDropdown();
        document.getElementById('result').textContent = 'Data loaded. Please select a movie.';
    } catch (error) {
        console.error('Initialization error:', error);
        // Error message is already displayed by loadData()
    }
};

// Populate the dropdown with movie titles
function populateMoviesDropdown() {
    const selectElement = document.getElementById('movie-select');
    
    // Clear the loading option
    selectElement.innerHTML = '';
    
    // Sort movies alphabetically by title
    const sortedMovies = [...movies].sort((a, b) => a.title.localeCompare(b.title));
    
    // Add a default option
    const defaultOption = document.createElement('option');
    defaultOption.value = '';
    defaultOption.textContent = 'Select a movie...';
    defaultOption.disabled = true;
    defaultOption.selected = true;
    selectElement.appendChild(defaultOption);
    
    // Add each movie as an option
    sortedMovies.forEach(movie => {
        const option = document.createElement('option');
        option.value = movie.id;
        option.textContent = movie.title;
        selectElement.appendChild(option);
    });
}

// Function to create a genre vector for a movie
function createGenreVector(movie, allGenres) {
    // Create a binary vector where 1 indicates the movie has the genre, 0 otherwise
    return allGenres.map(genre => movie.genres.includes(genre) ? 1 : 0);
}

// Function to calculate cosine similarity between two vectors
function cosineSimilarity(vecA, vecB) {
    // Calculate dot product
    let dotProduct = 0;
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
    }
    
    // Calculate magnitudes
    const magnitudeA = Math.sqrt(vecA.reduce((sum, val) => sum + val * val, 0));
    const magnitudeB = Math.sqrt(vecB.reduce((sum, val) => sum + val * val, 0));
    
    // Avoid division by zero
    if (magnitudeA === 0 || magnitudeB === 0) {
        return 0;
    }
    
    // Return cosine similarity
    return dotProduct / (magnitudeA * magnitudeB);
}

// Get all unique genres across all movies
function getAllGenres(movies) {
    const allGenres = new Set();
    movies.forEach(movie => {
        movie.genres.forEach(genre => {
            allGenres.add(genre);
        });
    });
    return Array.from(allGenres).sort();
}

// Main recommendation function
function getRecommendations() {
    // Step 1: Get user input
    const selectElement = document.getElementById('movie-select');
    const selectedMovieId = parseInt(selectElement.value);
    
    // Validate selection
    if (!selectedMovieId) {
        document.getElementById('result').textContent = 'Please select a movie first.';
        return;
    }
    
    // Step 2: Find the liked movie
    const likedMovie = movies.find(movie => movie.id === selectedMovieId);
    if (!likedMovie) {
        document.getElementById('result').textContent = 'Error: Selected movie not found.';
        return;
    }
    
    // Show loading message
    document.getElementById('result').textContent = 'Finding recommendations...';
    
    // Use setTimeout to allow the UI to update before the heavy computation
    setTimeout(() => {
        // Step 3: Get all unique genres and create genre vectors
        const allGenres = getAllGenres(movies);
        const likedMovieVector = createGenreVector(likedMovie, allGenres);
        
        // Step 4: Prepare candidate movies (exclude the liked movie)
        const candidateMovies = movies.filter(movie => movie.id !== likedMovie.id);
        
        // Step 5: Calculate cosine similarity scores
        const scoredMovies = candidateMovies.map(candidate => {
            const candidateVector = createGenreVector(candidate, allGenres);
            const score = cosineSimilarity(likedMovieVector, candidateVector);
            
            return {
                ...candidate,
                score: score
            };
        });
        
        // Step 6: Sort by score (descending)
        scoredMovies.sort((a, b) => b.score - a.score);
        
        // Step 7: Select top recommendations
        const topRecommendations = scoredMovies.slice(0, 2);
        
        // Step 8: Display results
        if (topRecommendations.length > 0) {
            const recommendationTitles = topRecommendations.map(movie => movie.title);
            const resultText = `Because you liked "${likedMovie.title}", we recommend: ${recommendationTitles.join(', ')}`;
            document.getElementById('result').textContent = resultText;
        } else {
            document.getElementById('result').textContent = `No recommendations found for "${likedMovie.title}".`;
        }
    }, 10); // Small delay to allow UI update
}
