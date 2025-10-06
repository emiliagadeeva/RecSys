class MovieLensApp {
    constructor() {
        this.data = {
            ratings: null,
            movies: null,
            users: null
        };
        this.models = {};
        this.isDataLoaded = false;
        this.isTraining = false;
        this.trainingHistories = {};
        this.currentTrainingModel = '';
        
        this.genreList = [
            'Unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 
            'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
            'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
            'Sci-Fi', 'Thriller', 'War', 'Western'
        ];
        
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        document.getElementById('loadData').addEventListener('click', () => this.loadData());
        document.getElementById('trainModels').addEventListener('click', () => this.trainAllModels());
        document.getElementById('compareModels').addEventListener('click', () => this.compareAllModels());
    }

    async loadData() {
        if (this.isTraining) {
            this.updateStatus('❌ Please wait for current operation to complete');
            return;
        }

        this.updateStatus('📥 Loading MovieLens 100K dataset...');
        this.showProgressBar();
        
        try {
            // Reset data
            this.data = { ratings: null, movies: null, users: null };
            this.isDataLoaded = false;

            await this.updateProgress(10, 'Loading movies data...');
            const moviesLoaded = await this.loadMovies();
            
            await this.updateProgress(40, 'Loading ratings data...');
            const ratingsLoaded = await this.loadRatings();
            
            await this.updateProgress(70, 'Loading user data...');
            const usersLoaded = await this.loadUsers();
            
            await this.updateProgress(100, 'Data processing completed!');

            if (moviesLoaded && ratingsLoaded && usersLoaded) {
                this.isDataLoaded = true;
                document.getElementById('trainModels').disabled = false;
                document.getElementById('compareModels').disabled = false;
                
                const stats = this.getDataStats();
                this.updateStatus(`✅ Data loaded: ${stats.ratings} ratings, ${stats.movies} movies, ${stats.users} users`);
            } else {
                throw new Error('Failed to load required data files');
            }
            
        } catch (error) {
            this.updateStatus('❌ Error loading data: ' + error.message);
            console.error('Data loading error:', error);
        } finally {
            this.hideProgressBar();
        }
    }

    getDataStats() {
        return {
            ratings: this.data.ratings ? this.data.ratings.length : 0,
            movies: this.data.movies ? this.data.movies.size : 0,
            users: this.data.users ? this.data.users.size : 0
        };
    }

    async loadMovies() {
        try {
            const response = await fetch('./data/u.item');
            if (!response.ok) {
                throw new Error(`Failed to load movies: ${response.status}`);
            }
            
            const text = await response.text();
            const movies = new Map();
            let loadedCount = 0;

            const lines = text.split('\n').filter(line => line.trim());
            for (const line of lines) {
                const parts = line.split('|');
                if (parts.length >= 24) {
                    const movieId = parseInt(parts[0]);
                    if (isNaN(movieId)) continue;

                    let title = parts[1];
                    // Clean title - remove year
                    title = title.replace(/\s*\(\d{4}\)\s*$/, '').trim();
                    
                    const genreVector = parts.slice(5, 24).map(x => {
                        const val = parseInt(x);
                        return isNaN(val) ? 0 : val;
                    });

                    movies.set(movieId, {
                        id: movieId,
                        title: title,
                        genres: genreVector,
                        genreNames: this.getGenreNames(genreVector),
                        releaseDate: parts[2] || 'Unknown'
                    });
                    loadedCount++;
                }
            }

            this.data.movies = movies;
            console.log(`✅ Loaded ${loadedCount} movies`);
            return true;

        } catch (error) {
            console.error('Failed to load real movies:', error);
            return await this.loadMockMovies();
        }
    }

    async loadRatings() {
        try {
            const response = await fetch('./data/u.data');
            if (!response.ok) {
                throw new Error(`Failed to load ratings: ${response.status}`);
            }

            const text = await response.text();
            const ratings = [];
            let loadedCount = 0;

            const lines = text.split('\n').filter(line => line.trim());
            for (const line of lines) {
                const parts = line.split('\t');
                if (parts.length >= 4) {
                    const userId = parseInt(parts[0]);
                    const movieId = parseInt(parts[1]);
                    const rating = parseInt(parts[2]);
                    
                    if (!isNaN(userId) && !isNaN(movieId) && !isNaN(rating)) {
                        ratings.push({
                            userId: userId,
                            movieId: movieId,
                            rating: rating,
                            timestamp: parseInt(parts[3]) || Date.now()
                        });
                        loadedCount++;
                    }
                }
            }

            this.data.ratings = ratings;
            console.log(`✅ Loaded ${loadedCount} ratings`);
            return true;

        } catch (error) {
            console.error('Failed to load real ratings:', error);
            return await this.loadMockRatings();
        }
    }

    async loadUsers() {
        try {
            const response = await fetch('./data/u.user');
            if (!response.ok) {
                throw new Error(`Failed to load users: ${response.status}`);
            }

            const text = await response.text();
            const users = new Map();
            let loadedCount = 0;

            const lines = text.split('\n').filter(line => line.trim());
            for (const line of lines) {
                const parts = line.split('|');
                if (parts.length >= 5) {
                    const userId = parseInt(parts[0]);
                    if (isNaN(userId)) continue;

                    users.set(userId, {
                        id: userId,
                        age: parseInt(parts[1]) || 25,
                        gender: parts[2] || 'U',
                        occupation: parts[3] || 'unknown',
                        zipCode: parts[4] || '00000'
                    });
                    loadedCount++;
                }
            }

            this.data.users = users;
            console.log(`✅ Loaded ${loadedCount} users`);
            return true;

        } catch (error) {
            console.error('Failed to load real users:', error);
            return await this.loadMockUsers();
        }
    }

    async loadMockMovies() {
        console.log('🔄 Creating mock movies data...');
        const movies = new Map();
        const mockTitles = [
            "The Matrix", "Inception", "Interstellar", "The Godfather", "Pulp Fiction",
            "Forrest Gump", "Fight Club", "The Shawshank Redemption", "The Dark Knight",
            "Star Wars", "Avatar", "Titanic", "Jurassic Park", "The Avengers", "Black Panther",
            "The Lion King", "Toy Story", "Frozen", "Spirited Away", "The Social Network"
        ];

        for (let i = 1; i <= 500; i++) {
            const title = i <= mockTitles.length ? mockTitles[i-1] : `Movie ${i}`;
            // Create more realistic genre distributions
            const genreVector = Array.from({length: 19}, (_, idx) => {
                // Some genres are more common
                const commonGenres = [1, 5, 7, 8, 14]; // Action, Comedy, Drama, Romance
                if (commonGenres.includes(idx)) return Math.random() > 0.6 ? 1 : 0;
                return Math.random() > 0.85 ? 1 : 0;
            });
            
            // Ensure at least one genre
            if (genreVector.every(v => v === 0)) {
                genreVector[Math.floor(Math.random() * 19)] = 1;
            }

            movies.set(i, {
                id: i,
                title: title,
                genres: genreVector,
                genreNames: this.getGenreNames(genreVector)
            });
        }

        this.data.movies = movies;
        console.log(`✅ Created ${movies.size} mock movies`);
        return true;
    }

    async loadMockRatings() {
        console.log('🔄 Creating mock ratings data...');
        const ratings = [];
        const numUsers = 200;
        const numMovies = 500;

        // Create user preferences for genres
        const userPreferences = new Map();
        for (let userId = 1; userId <= numUsers; userId++) {
            const preferredGenres = new Set();
            // Each user prefers 2-4 genres
            const numPreferred = Math.floor(Math.random() * 3) + 2;
            for (let i = 0; i < numPreferred; i++) {
                preferredGenres.add(Math.floor(Math.random() * 19));
            }
            userPreferences.set(userId, preferredGenres);
        }

        // Generate ratings based on preferences
        for (let i = 0; i < 5000; i++) {
            const userId = Math.floor(Math.random() * numUsers) + 1;
            const movieId = Math.floor(Math.random() * numMovies) + 1;
            const movie = this.data.movies.get(movieId);
            
            if (!movie) continue;

            let baseRating = 3; // Neutral base
            
            // Check genre match with user preferences
            const userPrefs = userPreferences.get(userId);
            let genreMatch = 0;
            movie.genres.forEach((hasGenre, genreIdx) => {
                if (hasGenre && userPrefs.has(genreIdx)) {
                    genreMatch++;
                }
            });

            // Adjust rating based on genre match
            if (genreMatch >= 2) baseRating += 1.5;
            else if (genreMatch >= 1) baseRating += 0.5;

            // Add some randomness
            const rating = Math.max(1, Math.min(5, baseRating + (Math.random() - 0.5) * 2));
            
            ratings.push({
                userId: userId,
                movieId: movieId,
                rating: Math.round(rating),
                timestamp: Date.now() - Math.floor(Math.random() * 1000000000)
            });
        }

        this.data.ratings = ratings;
        console.log(`✅ Created ${ratings.length} mock ratings`);
        return true;
    }

    async loadMockUsers() {
        console.log('🔄 Creating mock users data...');
        const users = new Map();
        const occupations = ['student', 'engineer', 'teacher', 'doctor', 'artist', 'scientist', 'writer'];
        const numUsers = 200;

        for (let i = 1; i <= numUsers; i++) {
            users.set(i, {
                id: i,
                age: Math.floor(Math.random() * 50) + 15,
                gender: Math.random() > 0.5 ? 'M' : 'F',
                occupation: occupations[Math.floor(Math.random() * occupations.length)],
                zipCode: String(10000 + Math.floor(Math.random() * 90000))
            });
        }

        this.data.users = users;
        console.log(`✅ Created ${users.size} mock users`);
        return true;
    }

    getGenreNames(genreVector) {
        return genreVector
            .map((value, index) => value === 1 ? this.genreList[index] : null)
            .filter(name => name !== null);
    }

    showProgressBar() {
        document.getElementById('progressBar').style.display = 'block';
        document.getElementById('progressFill').style.width = '0%';
    }

    hideProgressBar() {
        document.getElementById('progressBar').style.display = 'none';
    }

    async updateProgress(percent, message) {
        document.getElementById('progressFill').style.width = percent + '%';
        if (message) {
            this.updateStatus(message);
        }
        await new Promise(resolve => setTimeout(resolve, 100));
    }

    updateStatus(message) {
        const statusElement = document.getElementById('status');
        statusElement.textContent = message;
        console.log('Status:', message);
        
        // Auto-scroll to show latest status
        statusElement.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    async trainAllModels() {
    if (this.isTraining || !this.isDataLoaded) {
        this.updateStatus('❌ Please load data first');
        return;
    }
    
    this.isTraining = true;
    this.trainingHistories = {};
    
    try {
        // Clear previous charts
        document.getElementById('trainingCharts').innerHTML = '';
        
        // Get actual data dimensions from loaded data
        const numUsers = this.data.users.size;
        const numMovies = this.data.movies.size;
        const numGenres = 19;
        
        console.log(`Data dimensions - Users: ${numUsers}, Movies: ${numMovies}, Genres: ${numGenres}`);
        
        // Initialize models with correct dimensions
        this.models = {
            baseline: new WithoutDLTwoTower(32, numUsers, numMovies, numGenres),
            mlp: new MLPTwoTower(32, numUsers, numMovies, numGenres),
            deep: new DeepLearningTwoTower(32, numUsers, numMovies, numGenres)
        };

        // Set up status callbacks
        Object.entries(this.models).forEach(([name, model]) => {
            model.setStatusCallback((msg) => {
                this.updateStatus(`[${name.toUpperCase()}] ${msg}`);
            });
        });

        // Prepare training data
        this.updateStatus('📊 Preparing training data...');
        const trainingData = this.prepareTrainingData();
        
        if (trainingData.userInput.shape[0] === 0) {
            throw new Error('No training data available');
        }

        console.log(`Training with ${trainingData.userInput.shape[0]} samples`);

        // Train models sequentially
        const trainingConfigs = [
            { name: 'baseline', displayName: 'Without Deep Learning' },
            { name: 'mlp', displayName: 'MLP Two-Tower' },
            { name: 'deep', displayName: 'Deep Learning Two-Tower' }
        ];

        for (const config of trainingConfigs) {
            this.currentTrainingModel = config.displayName;
            this.updateStatus(`⚡ Training ${config.displayName}...`);
            
            console.log(`=== Starting training for ${config.name} ===`);
            this.trainingHistories[config.name] = await this.models[config.name].train(
                trainingData.userInput,
                trainingData.movieInput,
                trainingData.ratings,
                5, // epochs
                32  // batchSize
            );
            
            console.log(`=== Completed training for ${config.name} ===`);
            this.updateStatus(`✅ ${config.displayName} training completed`);
            
            // Small delay between models
            await new Promise(resolve => setTimeout(resolve, 500));
        }

        // Plot results
        this.plotTrainingHistories();
        this.updateStatus('🎉 All models trained successfully! Click "Compare All Models" to see recommendations.');

    } catch (error) {
        this.updateStatus('❌ Training failed: ' + error.message);
        console.error('Training error:', error);
    } finally {
        this.isTraining = false;
        this.currentTrainingModel = '';
    }
}
    prepareTrainingData() {
        console.log('Preparing training data...');
        
        // Get positive interactions (ratings >= 4)
        const positiveInteractions = this.data.ratings.filter(r => r.rating >= 4);
        console.log(`Found ${positiveInteractions.length} positive interactions`);

        // Create negative samples
        const negativeInteractions = this.createNegativeSamples(positiveInteractions.length);
        console.log(`Created ${negativeInteractions.length} negative samples`);

        // Combine and shuffle
        const allInteractions = [
            ...positiveInteractions.map(r => ({ ...r, label: 1 })),
            ...negativeInteractions.map(r => ({ ...r, label: 0 }))
        ];

        this.shuffleArray(allInteractions);
        console.log(`Total training samples: ${allInteractions.length}`);

        // Prepare tensors
        const userInput = allInteractions.map(r => {
            const userId = r.userId - 1; // Convert to 0-based
            return Math.max(0, Math.min(199, userId)); // Ensure within bounds
        });

        const movieInput = allInteractions.map(r => {
            const movie = this.data.movies.get(r.movieId);
            if (!movie) {
                console.warn(`Movie ${r.movieId} not found, using default`);
                return { movieId: 0, genres: Array(19).fill(0) };
            }
            return {
                movieId: Math.max(0, Math.min(499, r.movieId - 1)), // 0-based, bounded
                genres: movie.genres
            };
        });

        const ratings = allInteractions.map(r => r.label);

        // Validate data
        const validIndices = userInput.map((userId, idx) => 
            !isNaN(userId) && userId >= 0 && userId < 200 && 
            !isNaN(movieInput[idx].movieId) && movieInput[idx].movieId >= 0 && movieInput[idx].movieId < 500
        ).filter(valid => valid).length;

        console.log(`Valid samples: ${validIndices}/${userInput.length}`);

        return {
            userInput: tf.tensor1d(userInput, 'int32'),
            movieInput: movieInput,
            ratings: tf.tensor1d(ratings, 'float32')
        };
    }

    createNegativeSamples(targetCount) {
        console.log('Creating negative samples...');
        const negativeSamples = [];
        const userMovies = new Map();

        // Build user-movie interaction map
        this.data.ratings.forEach(rating => {
            if (!userMovies.has(rating.userId)) {
                userMovies.set(rating.userId, new Set());
            }
            userMovies.get(rating.userId).add(rating.movieId);
        });

        const allMovieIds = Array.from(this.data.movies.keys());
        const maxAttempts = targetCount * 5;
        let attempts = 0;
        let created = 0;

        while (created < targetCount && attempts < maxAttempts) {
            const randomUser = Math.floor(Math.random() * 200) + 1;
            const randomMovie = allMovieIds[Math.floor(Math.random() * allMovieIds.length)];

            // Check if user hasn't watched this movie
            if (!userMovies.get(randomUser)?.has(randomMovie)) {
                negativeSamples.push({
                    userId: randomUser,
                    movieId: randomMovie,
                    label: 0
                });
                created++;
            }
            attempts++;
        }

        console.log(`Created ${negativeSamples.length} negative samples after ${attempts} attempts`);
        return negativeSamples;
    }

    shuffleArray(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
    }

    plotTrainingHistories() {
        console.log('Plotting training histories...');
        
        const histories = [
            { name: 'Without DL', history: this.trainingHistories.baseline, color: '#007cba' },
            { name: 'MLP Two-Tower', history: this.trainingHistories.mlp, color: '#28a745' },
            { name: 'Deep Two-Tower', history: this.trainingHistories.deep, color: '#dc3545' }
        ].filter(model => model.history && model.history.loss && model.history.loss.length > 0);

        if (histories.length === 0) {
            console.warn('No valid training histories to plot');
            this.updateStatus('⚠️ No training data to display in charts');
            return;
        }

        console.log(`Plotting ${histories.length} model histories`);

        const lossSeries = histories.map(model => ({
            values: model.history.loss,
            name: model.name
        }));

        const accuracySeries = histories.map(model => ({
            values: model.history.acc || Array(model.history.loss.length).fill(0.5),
            name: model.name
        }));

        const chartsContainer = document.getElementById('trainingCharts');
        chartsContainer.innerHTML = '';

        // Create container for charts
        const chartsDiv = document.createElement('div');
        chartsDiv.style.display = 'flex';
        chartsDiv.style.flexDirection = 'column';
        chartsDiv.style.gap = '20px';
        chartsDiv.style.alignItems = 'center';

        // Loss chart
        const lossHeader = document.createElement('h3');
        lossHeader.textContent = '📉 Training Loss';
        chartsDiv.appendChild(lossHeader);

        const lossContainer = document.createElement('div');
        lossContainer.id = 'lossChart';
        chartsDiv.appendChild(lossContainer);

        // Accuracy chart
        const accHeader = document.createElement('h3');
        accHeader.textContent = '📈 Training Accuracy';
        chartsDiv.appendChild(accHeader);

        const accContainer = document.createElement('div');
        accContainer.id = 'accChart';
        chartsDiv.appendChild(accContainer);

        chartsContainer.appendChild(chartsDiv);

        try {
            // Plot loss
            tfvis.show.history(
                { name: 'Training Loss', tab: 'Loss' },
                lossSeries,
                ['line', 'line', 'line'].slice(0, histories.length),
                {
                    xLabel: 'Epoch',
                    yLabel: 'Loss',
                    width: 500,
                    height: 300
                },
                lossContainer
            );

            // Plot accuracy
            tfvis.show.history(
                { name: 'Training Accuracy', tab: 'Accuracy' },
                accuracySeries,
                ['line', 'line', 'line'].slice(0, histories.length),
                {
                    xLabel: 'Epoch',
                    yLabel: 'Accuracy',
                    width: 500,
                    height: 300
                },
                accContainer
            );

            console.log('Charts plotted successfully');

        } catch (error) {
            console.error('Error plotting charts:', error);
            chartsContainer.innerHTML = '<p>Error displaying training charts. Check console for details.</p>';
        }
    }

    async compareAllModels() {
        if (!this.models.baseline || !this.models.mlp || !this.models.deep) {
            this.updateStatus('❌ Please train all models first');
            return;
        }

        this.updateStatus('📊 Comparing all three models...');

        try {
            const testUserId = this.findUserWithRatings();
            if (!testUserId) {
                this.updateStatus('❌ No suitable test user found with sufficient ratings');
                return;
            }

            this.updateStatus(`🧪 Testing with user ${testUserId}...`);

            // Get user history
            const userHistory = this.getUserHistory(testUserId);
            this.displayUserHistory(userHistory);

            // Get recommendations from all models
            const [baselineRecs, mlpRecs, deepRecs] = await Promise.all([
                this.getRecommendations(this.models.baseline, testUserId, 10),
                this.getRecommendations(this.models.mlp, testUserId, 10),
                this.getRecommendations(this.models.deep, testUserId, 10)
            ]);

            // Display recommendations
            this.displayRecommendations(baselineRecs, mlpRecs, deepRecs);

            // Calculate metrics
            const metrics = await this.calculateAllMetrics(testUserId);
            this.displayMetrics(metrics);

            // Show comparison section
            document.getElementById('comparisonSection').style.display = 'block';
            this.updateStatus('✅ Model comparison completed! Scroll down to see results.');

        } catch (error) {
            this.updateStatus('❌ Error comparing models: ' + error.message);
            console.error('Comparison error:', error);
        }
    }

    getUserHistory(userId) {
        return this.data.ratings
            .filter(r => r.userId === userId)
            .sort((a, b) => b.rating - a.rating)
            .slice(0, 10)
            .map(r => {
                const movie = this.data.movies.get(r.movieId);
                return {
                    title: movie ? movie.title : `Movie ${r.movieId}`,
                    genres: movie ? movie.genreNames : ['Unknown'],
                    rating: r.rating
                };
            });
    }

    displayUserHistory(history) {
        const tbody = document.querySelector('#historyTable tbody');
        tbody.innerHTML = '';

        if (history.length === 0) {
            const row = tbody.insertRow();
            const cell = row.insertCell(0);
            cell.colSpan = 3;
            cell.textContent = 'No rating history available';
            cell.style.textAlign = 'center';
            cell.style.color = '#666';
            cell.style.padding = '20px';
            return;
        }

        history.forEach(item => {
            const row = tbody.insertRow();
            
            // Title
            const titleCell = row.insertCell(0);
            titleCell.textContent = item.title;
            titleCell.style.maxWidth = '200px';
            titleCell.style.overflow = 'hidden';
            titleCell.style.textOverflow = 'ellipsis';
            titleCell.style.whiteSpace = 'nowrap';

            // Genres
            const genreCell = row.insertCell(1);
            if (item.genres && item.genres.length > 0) {
                item.genres.forEach(genre => {
                    const span = document.createElement('span');
                    span.className = 'genre-tag';
                    span.textContent = genre;
                    genreCell.appendChild(span);
                });
            } else {
                genreCell.textContent = 'Unknown';
                genreCell.style.color = '#999';
            }

            // Rating
            const ratingCell = row.insertCell(2);
            ratingCell.textContent = '⭐'.repeat(item.rating);
            ratingCell.title = `Rating: ${item.rating}/5`;
        });
    }

    findUserWithRatings() {
        // Find user with at least 5 high ratings
        const userRatingCounts = new Map();
        
        this.data.ratings.forEach(rating => {
            if (rating.rating >= 4) {
                userRatingCounts.set(rating.userId, (userRatingCounts.get(rating.userId) || 0) + 1);
            }
        });

        // Find user with most high ratings
        let bestUser = null;
        let maxRatings = 0;
        
        for (const [userId, count] of userRatingCounts) {
            if (count > maxRatings) {
                maxRatings = count;
                bestUser = userId;
            }
        }

        console.log(`Selected user ${bestUser} with ${maxRatings} high ratings`);
        return bestUser;
    }

    async getRecommendations(model, userId, count = 10) {
        try {
            this.updateStatus(`🔍 ${model.constructor.name} generating recommendations...`);
            
            const userEmbedding = await model.getUserEmbedding(userId - 1);
            const allMovies = Array.from(this.data.movies.values());
            const scores = [];

            // Test with subset for performance (first 200 movies)
            const testMovies = allMovies.slice(0, 200);
            
            for (const movie of testMovies) {
                try {
                    const movieEmbedding = await model.getItemEmbedding(movie.id - 1, movie.genres);
                    const score = await this.calculateSimilarity(userEmbedding, movieEmbedding);
                    scores.push({ movie, score });
                    movieEmbedding.dispose();
                } catch (error) {
                    console.warn(`Error processing movie ${movie.id}:`, error);
                }
            }

            userEmbedding.dispose();

            const recommendations = scores
                .sort((a, b) => b.score - a.score)
                .slice(0, count)
                .map(item => ({
                    title: item.movie.title,
                    genres: item.movie.genreNames,
                    score: item.score
                }));

            console.log(`${model.constructor.name} recommendations:`, recommendations);
            return recommendations;

        } catch (error) {
            console.error('Error generating recommendations:', error);
            return [];
        }
    }

    async calculateSimilarity(vec1, vec2) {
        try {
            // Use cosine similarity for better results
            const dotProduct = vec1.mul(vec2).sum();
            const norm1 = vec1.norm();
            const norm2 = vec2.norm();
            const similarity = dotProduct.div(norm1.mul(norm2));
            const result = await similarity.data();
            
            tf.dispose([dotProduct, norm1, norm2, similarity]);
            
            // Convert from [-1, 1] to [0, 1] range
            return (result[0] + 1) / 2;
        } catch (error) {
            // Fallback to dot product
            const dotProduct = vec1.mul(vec2).sum();
            const result = await dotProduct.data();
            dotProduct.dispose();
            return Math.max(0, result[0]);
        }
    }

    displayRecommendations(baselineRecs, mlpRecs, deepRecs) {
        this.populateRecommendationTable('baselineRecommendations', baselineRecs);
        this.populateRecommendationTable('mlpRecommendations', mlpRecs);
        this.populateRecommendationTable('deepRecommendations', deepRecs);
    }

    populateRecommendationTable(tableId, recommendations) {
        const tbody = document.getElementById(tableId);
        tbody.innerHTML = '';

        if (recommendations.length === 0) {
            const row = tbody.insertRow();
            const cell = row.insertCell(0);
            cell.colSpan = 3;
            cell.textContent = 'No recommendations generated';
            cell.style.textAlign = 'center';
            cell.style.color = '#666';
            cell.style.padding = '20px';
            return;
        }

        recommendations.forEach(item => {
            const row = tbody.insertRow();
            
            // Title
            const titleCell = row.insertCell(0);
            titleCell.textContent = item.title;
            titleCell.style.maxWidth = '150px';
            titleCell.style.overflow = 'hidden';
            titleCell.style.textOverflow = 'ellipsis';
            titleCell.style.whiteSpace = 'nowrap';

            // Genres
            const genreCell = row.insertCell(1);
            if (item.genres && item.genres.length > 0) {
                item.genres.forEach(genre => {
                    const span = document.createElement('span');
                    span.className = 'genre-tag';
                    span.textContent = genre;
                    genreCell.appendChild(span);
                });
            } else {
                genreCell.textContent = 'Unknown';
                genreCell.style.color = '#999';
            }

            // Score with color coding
            const scoreCell = row.insertCell(2);
            scoreCell.textContent = item.score.toFixed(4);
            scoreCell.className = 'score-cell';
            scoreCell.style.fontFamily = 'monospace';
            
            // Color based on score
            if (item.score > 0.7) {
                scoreCell.classList.add('high-score');
            } else if (item.score > 0.4) {
                scoreCell.classList.add('medium-score');
            } else {
                scoreCell.classList.add('low-score');
            }
        });
    }

    async calculateAllMetrics(testUserId) {
        const k = 5;
        const userRatings = this.data.ratings.filter(r => r.userId === testUserId && r.rating >= 4);
        const relevantItems = new Set(userRatings.map(r => r.movieId));
        
        const metrics = {};

        for (const [modelName, model] of Object.entries(this.models)) {
            try {
                const recommendations = await this.getRecommendations(model, testUserId, k);
                let hits = 0;

                recommendations.forEach(rec => {
                    const movie = Array.from(this.data.movies.values()).find(m => m.title === rec.title);
                    if (movie && relevantItems.has(movie.id)) {
                        hits++;
                    }
                });

                const precision = hits / k;
                const recall = userRatings.length > 0 ? hits / Math.min(userRatings.length, k) : 0;
                const f1 = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0;

                metrics[modelName] = { precision, recall, f1 };
                
                console.log(`${modelName} metrics:`, { precision, recall, f1, hits });

            } catch (error) {
                console.warn(`Error calculating metrics for ${modelName}:`, error);
                metrics[modelName] = { precision: 0, recall: 0, f1: 0 };
            }
        }

        return metrics;
    }

    displayMetrics(metrics) {
        const metricsPanel = document.getElementById('metricsPanel');
        
        const metricCards = {
            baseline: { name: '📈 Without DL', class: 'baseline' },
            mlp: { name: '🔬 MLP Two-Tower', class: 'mlp' },
            deep: { name: '🧠 Deep Two-Tower', class: 'deep' }
        };
        
        metricsPanel.innerHTML = Object.entries(metricCards).map(([key, config]) => {
            const metric = metrics[key] || { precision: 0, recall: 0, f1: 0 };
            return `
                <div class="metric-card ${config.class}">
                    <h4>${config.name}</h4>
                    <div class="metric-value">${metric.precision.toFixed(3)}</div>
                    <div><strong>Precision@5</strong></div>
                    <div>Recall@5: ${metric.recall.toFixed(3)}</div>
                    <div>F1 Score: ${metric.f1.toFixed(3)}</div>
                </div>
            `;
        }).join('');
    }
}

// Initialize application
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new MovieLensApp();
    console.log('🎬 MovieLens Three-Model Comparison Demo initialized');
});
