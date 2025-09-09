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
        // Step 3: Prepare for similarity calculation
        const likedGenres = new Set(likedMovie.genres);
        const candidateMovies = movies.filter(movie => movie.id !== likedMovie.id);
        
        // Step 4: Calculate Jaccard similarity scores
        const scoredMovies = candidateMovies.map(candidate => {
            const candidateGenres = new Set(candidate.genres);
            
            // Calculate intersection
            const intersection = new Set(
                [...likedGenres].filter(genre => candidateGenres.has(genre))
            );
            
            // Calculate union
            const union = new Set([...likedGenres, ...candidateGenres]);
            
            // Calculate Jaccard index
            const score = union.size > 0 ? intersection.size / union.size : 0;
            
            return {
                ...candidate,
                score: score
            };
        });
        
        // Step 5: Sort by score (descending)
        scoredMovies.sort((a, b) => b.score - a.score);
        
        // Step 6: Select top recommendations
        const topRecommendations = scoredMovies.slice(0, 2);
        
        // Step 7: Display results
        if (topRecommendations.length > 0) {
            const recommendationTitles = topRecommendations.map(movie => movie.title);
            const resultText = `Because you liked "${likedMovie.title}", we recommend: ${recommendationTitles.join(', ')}`;
            document.getElementById('result').textContent = resultText;
        } else {
            document.getElementById('result').textContent = `No recommendations found for "${likedMovie.title}".`;
        }
    }, 10); // Small delay to allow UI update
}
