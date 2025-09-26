// Global variables to store parsed data
let movies = [];       // Array of movie objects: {id, title, ...}
let ratings = [];      // Array of rating objects: {userId, movieId, rating}
let numUsers = 0;      // Number of unique users
let numMovies = 0;     // Number of unique movies

/**
 * Main function to load and parse all data files
 * This function coordinates the loading of both movie metadata and ratings data
 */
async function loadData() {
    try {
        console.log('Starting data loading process...');
        
        // Load and parse movie metadata first
        const movieDataUrl = 'https://raw.githubusercontent.com/sidooms/MovieTweetings/master/latest/movies.dat';
        console.log('Loading movie data from:', movieDataUrl);
        const movieResponse = await fetch(movieDataUrl);
        const movieText = await movieResponse.text();
        movies = parseItemData(movieText);
        numMovies = movies.length;
        console.log(`Parsed ${numMovies} movies`);
        
        // Load and parse ratings data
        const ratingDataUrl = 'https://raw.githubusercontent.com/sidooms/MovieTweetings/master/latest/ratings.dat';
        console.log('Loading rating data from:', ratingDataUrl);
        const ratingResponse = await fetch(ratingDataUrl);
        const ratingText = await ratingResponse.text();
        ratings = parseRatingData(ratingText);
        
        // Calculate number of unique users
        const uniqueUsers = new Set(ratings.map(r => r.userId));
        numUsers = uniqueUsers.size;
        
        console.log(`Data loading completed:`);
        console.log(`- Users: ${numUsers}`);
        console.log(`- Movies: ${numMovies}`);
        console.log(`- Ratings: ${ratings.length}`);
        
        return { movies, ratings, numUsers, numMovies };
        
    } catch (error) {
        console.error('Error loading data:', error);
        throw new Error(`Failed to load data: ${error.message}`);
    }
}

/**
 * Parse movie metadata from text format
 * Expected format: MovieID::Title::Genres
 * Example: 1::Toy Story (1995)::Animation|Children's|Comedy
 * 
 * Why we do this: We need to map movie IDs to human-readable titles for the UI
 * and maintain consistency between movie IDs in ratings and movie metadata.
 */
function parseItemData(text) {
    console.log('Parsing movie data...');
    const lines = text.split('\n');
    const movieList = [];
    
    for (const line of lines) {
        if (line.trim() === '') continue;
        
        try {
            // Handle different possible separators
            const parts = line.split('::');
            if (parts.length >= 2) {
                const movieId = parseInt(parts[0]);
                // Extract year from title if present (format: "Title (Year)")
                const title = parts[1].trim();
                
                movieList.push({
                    id: movieId,
                    title: title,
                    genres: parts[2] ? parts[2].split('|') : []
                });
            }
        } catch (error) {
            console.warn('Skipping invalid movie line:', line.substring(0, 50));
        }
    }
    
    // Sort by movie ID for consistency
    movieList.sort((a, b) => a.id - b.id);
    console.log(`Successfully parsed ${movieList.length} movies`);
    return movieList;
}

/**
 * Parse ratings data from text format
 * Expected format: UserID::MovieID::Rating::Timestamp
 * Example: 196::242::3::881250949
 * 
 * Why we do this: Ratings are the core training data for our matrix factorization model.
 * We need to convert this text data into numerical arrays that TensorFlow.js can process.
 * We also normalize user and movie IDs to be 0-indexed for the embedding layers.
 */
function parseRatingData(text) {
    console.log('Parsing rating data...');
    const lines = text.split('\n');
    const ratingList = [];
    let minUserId = Infinity;
    let minMovieId = Infinity;
    
    // First pass: find min IDs to normalize to 0-indexed
    for (const line of lines) {
        if (line.trim() === '') continue;
        
        const parts = line.split('::');
        if (parts.length >= 3) {
            const userId = parseInt(parts[0]);
            const movieId = parseInt(parts[1]);
            
            minUserId = Math.min(minUserId, userId);
            minMovieId = Math.min(minMovieId, movieId);
        }
    }
    
    // Second pass: parse all ratings with normalized IDs
    for (const line of lines) {
        if (line.trim() === '') continue;
        
        try {
            const parts = line.split('::');
            if (parts.length >= 3) {
                // Normalize IDs to be 0-indexed for TensorFlow embeddings
                const userId = parseInt(parts[0]) - minUserId;
                const movieId = parseInt(parts[1]) - minMovieId;
                const rating = parseFloat(parts[2]);
                
                // Only include valid ratings
                if (!isNaN(userId) && !isNaN(movieId) && !isNaN(rating) && rating >= 0.5 && rating <= 5) {
                    ratingList.push({
                        userId: userId,
                        movieId: movieId,
                        rating: rating,
                        timestamp: parts[3] ? parseInt(parts[3]) : 0
                    });
                }
            }
        } catch (error) {
            console.warn('Skipping invalid rating line:', line.substring(0, 50));
        }
    }
    
    console.log(`Successfully parsed ${ratingList.length} ratings`);
    console.log(`User ID range: ${Math.min(...ratingList.map(r => r.userId))} - ${Math.max(...ratingList.map(r => r.userId))}`);
    console.log(`Movie ID range: ${Math.min(...ratingList.map(r => r.movieId))} - ${Math.max(...ratingList.map(r => r.movieId))}`);
    
    return ratingList;
}
