// Global variables to store parsed data
let movies = [];
let ratings = [];
let numUsers = 0;
let numMovies = 0;

/**
 * Load MovieLens 100K data from local files u.item and u.data
 */
async function loadData() {
    try {
        console.log('Starting data loading process...');
        updateResult('Loading movie data from u.item...', 'info');
        
        // Load movie data from local u.item file
        const movieResponse = await fetch('u.item');
        if (!movieResponse.ok) {
            throw new Error(`Failed to load u.item: ${movieResponse.status}`);
        }
        const movieText = await movieResponse.text();
        movies = parseItemData(movieText);
        numMovies = movies.length;
        
        updateResult(`Loaded ${numMovies} movies. Loading ratings from u.data...`, 'info');
        
        // Load ratings data from local u.data file
        const ratingResponse = await fetch('u.data');
        if (!ratingResponse.ok) {
            throw new Error(`Failed to load u.data: ${ratingResponse.status}`);
        }
        const ratingText = await ratingResponse.text();
        ratings = parseRatingData(ratingText);
        
        // Calculate number of unique users
        const uniqueUsers = new Set(ratings.map(r => r.userId));
        numUsers = uniqueUsers.size;
        
        console.log('Data loading completed:');
        console.log('- Users:', numUsers);
        console.log('- Movies:', numMovies);
        console.log('- Ratings:', ratings.length);
        
        updateResult(`Data loaded: ${numUsers} users, ${numMovies} movies, ${ratings.length} ratings`, 'success');
        return { movies, ratings, numUsers, numMovies };
        
    } catch (error) {
        console.error('Error loading data:', error);
        updateResult(`Error loading local files: ${error.message}. Using demo data...`, 'error');
        // Fallback to demo data
        await createDemoData();
        throw error;
    }
}

/**
 * Parse u.item file - MovieLens 100K format
 * Format: movieId|movieTitle|releaseDate|videoReleaseDate|IMDbURL|...|genres
 * Example: 1|Toy Story (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)|0|0|0|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0
 */
function parseItemData(text) {
    console.log('Parsing u.item movie data...');
    const lines = text.split('\n');
    const movieList = [];
    let lineCount = 0;
    
    for (const line of lines) {
        if (line.trim() === '') continue;
        lineCount++;
        
        try {
            const parts = line.split('|');
            if (parts.length >= 2) {
                const movieId = parseInt(parts[0]);
                if (isNaN(movieId)) continue;
                
                // Extract title and year (format: "Title (Year)")
                const title = parts[1].trim();
                const yearMatch = title.match(/.+\((\d{4})\)/);
                const year = yearMatch ? parseInt(yearMatch[1]) : null;
                
                // Parse genres (last 19 fields)
                const genres = [];
                const genreNames = [
                    "Unknown", "Action", "Adventure", "Animation", "Children's", 
                    "Comedy", "Crime", "Documentary", "Drama", "Fantasy", 
                    "Film-Noir", "Horror", "Musical", "Mystery", "Romance", 
                    "Sci-Fi", "Thriller", "War", "Western"
                ];
                
                for (let i = 5; i <= 23; i++) {
                    if (parts[i] === '1') {
                        genres.push(genreNames[i - 5]);
                    }
                }
                
                movieList.push({
                    id: movieId,
                    title: title,
                    year: year,
                    genres: genres
                });
            }
        } catch (error) {
            console.warn('Skipping invalid movie line:', line.substring(0, 50));
        }
        
        // Limit for performance if file is large
        if (lineCount >= 2000) break;
    }
    
    console.log(`Successfully parsed ${movieList.length} movies`);
    return movieList;
}

/**
 * Parse u.data file - MovieLens 100K format  
 * Format: userId\tmovieId\trating\ttimestamp
 * Example: 196\t242\t3\t881250949
 */
function parseRatingData(text) {
    console.log('Parsing u.data rating data...');
    const lines = text.split('\n');
    const ratingList = [];
    let lineCount = 0;
    
    // Find minimum user ID and movie ID to normalize to 0-indexed
    let minUserId = Infinity;
    let minMovieId = Infinity;
    
    // First pass: find min IDs
    for (const line of lines) {
        if (line.trim() === '') continue;
        const parts = line.split('\t');
        if (parts.length >= 3) {
            minUserId = Math.min(minUserId, parseInt(parts[0]));
            minMovieId = Math.min(minMovieId, parseInt(parts[1]));
        }
    }
    
    // Second pass: parse all ratings with normalized IDs
    for (const line of lines) {
        if (line.trim() === '') continue;
        lineCount++;
        
        try {
            const parts = line.split('\t');
            if (parts.length >= 3) {
                // Normalize IDs to be 0-indexed for TensorFlow embeddings
                const userId = parseInt(parts[0]) - minUserId;
                const movieId = parseInt(parts[1]) - minMovieId;
                const rating = parseFloat(parts[2]);
                
                if (!isNaN(userId) && !isNaN(movieId) && !isNaN(rating) && 
                    rating >= 0.5 && rating <= 5 && userId >= 0 && movieId >= 0) {
                    ratingList.push({
                        userId: userId,
                        movieId: movieId,
                        rating: rating,
                        timestamp: parts[3] ? parseInt(parts[3]) : 0
                    });
                }
            }
        } catch (error) {
            console.warn('Skipping invalid rating line');
        }
        
        // Limit for performance
        if (lineCount >= 10000) break;
    }
    
    console.log(`Successfully parsed ${ratingList.length} ratings`);
    return ratingList;
}

/**
 * Fallback demo data creation if local files are not available
 */
async function createDemoData() {
    console.log('Creating demo data...');
    updateResult('Local files not found. Using demo data...', 'warning');
    
    // Create minimal demo data
    movies = [
        { id: 1, title: "Toy Story (1995)", year: 1995, genres: ["Animation", "Comedy"] },
        { id: 2, title: "Jumanji (1995)", year: 1995, genres: ["Adventure", "Fantasy"] },
        { id: 3, title: "Grumpier Old Men (1995)", year: 1995, genres: ["Comedy", "Romance"] },
        { id: 4, title: "Waiting to Exhale (1995)", year: 1995, genres: ["Comedy", "Drama"] },
        { id: 5, title: "Father of the Bride Part II (1995)", year: 1995, genres: ["Comedy"] }
    ];
    
    ratings = [
        { userId: 1, movieId: 1, rating: 4.0 },
        { userId: 1, movieId: 2, rating: 3.0 },
        { userId: 1, movieId: 3, rating: 5.0 },
        { userId: 2, movieId: 1, rating: 5.0 },
        { userId: 2, movieId: 4, rating: 4.0 },
        { userId: 3, movieId: 5, rating: 3.5 },
        { userId: 3, movieId: 2, rating: 4.5 }
    ];
    
    numUsers = 3;
    numMovies = 5;
    
    console.log('Demo data created:', { numUsers, numMovies, ratings: ratings.length });
}

// Helper function to update UI from data.js
function updateResult(message, type) {
    const resultElement = document.getElementById('result');
    if (resultElement) {
        const colors = { info: '#3498db', success: '#27ae60', error: '#e74c3c', warning: '#f39c12' };
        resultElement.style.borderLeftColor = colors[type] || colors.info;
        resultElement.innerHTML = `<p>${message}</p>`;
    }
}
