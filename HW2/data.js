// Global variables for movie and rating data
let movies = [];
let ratings = [];

// Genre names in the order they appear in the u.item file
const genres = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", 
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", 
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
];

// Main function to load data from files
async function loadData() {
    try {
        // Load and parse movie data
        const moviesResponse = await fetch('u.item');
        if (!moviesResponse.ok) {
            throw new Error(`Failed to load movie data: ${moviesResponse.status}`);
        }
        const moviesText = await moviesResponse.text();
        parseItemData(moviesText);
        
        // Load and parse rating data
        const ratingsResponse = await fetch('u.data');
        if (!ratingsResponse.ok) {
            throw new Error(`Failed to load rating data: ${ratingsResponse.status}`);
        }
        const ratingsText = await ratingsResponse.text();
        parseRatingData(ratingsText);
        
        console.log('Data loaded successfully');
    } catch (error) {
        console.error('Error loading data:', error);
        const resultElement = document.getElementById('result');
        if (resultElement) {
            resultElement.textContent = `Error loading data: ${error.message}. Please make sure u.item and u.data files are in the correct location.`;
        }
        throw error; // Re-throw to allow script.js to handle the error
    }
}

// Parse movie data from u.item format
function parseItemData(text) {
    const lines = text.split('\n');
    
    for (const line of lines) {
        if (line.trim() === '') continue;
        
        const fields = line.split('|');
        if (fields.length < 5) continue;
        
        const id = parseInt(fields[0]);
        const title = fields[1];
        const genreIndicators = fields.slice(5, 24); // Get the 19 genre indicators
        
        // Convert genre indicators to an array of genre names
        const movieGenres = genreIndicators.map((indicator, index) => {
            return indicator === '1' ? genres[index] : null;
        }).filter(genre => genre !== null);
        
        movies.push({
            id,
            title,
            genres: movieGenres
        });
    }
}

// Parse rating data from u.data format
function parseRatingData(text) {
    const lines = text.split('\n');
    
    for (const line of lines) {
        if (line.trim() === '') continue;
        
        const fields = line.split('\t');
        if (fields.length < 4) continue;
        
        ratings.push({
            userId: parseInt(fields[0]),
            itemId: parseInt(fields[1]),
            rating: parseInt(fields[2]),
            timestamp: parseInt(fields[3])
        });
    }
}
