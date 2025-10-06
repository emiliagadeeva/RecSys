class BaseTwoTower {
    constructor(embeddingDim, numUsers, numItems, numGenres) {
        this.embeddingDim = embeddingDim;
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.numGenres = numGenres;
        this.model = null;
        this.userTower = null;
        this.itemTower = null;
    }

    buildModel() {
        throw new Error('buildModel must be implemented by subclass');
    }

    async train(userInput, movieInput, ratings, epochs = 3, batchSize = 128) {
        if (!this.model) {
            this.buildModel();
        }

        console.log(`Starting training for ${this.constructor.name}`);
        const history = {
            loss: [],
            acc: []
        };

        // Convert to tensors once
        const movieIds = tf.tensor1d(movieInput.map(m => m.movieId), 'int32');
        const genres = tf.tensor2d(movieInput.map(m => m.genres));

        for (let epoch = 0; epoch < epochs; epoch++) {
            const epochLoss = tf.tensor1d([0]);
            const epochAcc = tf.tensor1d([0]);
            let batchCount = 0;

            const numSamples = userInput.shape[0];
            
            for (let i = 0; i < numSamples; i += batchSize) {
                const end = Math.min(i + batchSize, numSamples);
                
                const userBatch = userInput.slice([i], [end - i]);
                const movieIdBatch = movieIds.slice([i], [end - i]);
                const genreBatch = genres.slice([i, 0], [end - i, -1]);
                const labelBatch = ratings.slice([i], [end - i]);

                try {
                    const result = await this.model.trainOnBatch(
                        [userBatch, movieIdBatch, genreBatch], 
                        labelBatch
                    );

                    // Update metrics
                    const loss = tf.tensor1d([result[0]]);
                    const acc = tf.tensor1d([result[1]]);
                    
                    const newLoss = epochLoss.add(loss);
                    const newAcc = epochAcc.add(acc);
                    
                    epochLoss.dispose();
                    epochAcc.dispose();
                    
                    epochLoss = newLoss;
                    epochAcc = newAcc;
                    batchCount++;

                    // Cleanup
                    tf.dispose([loss, acc, userBatch, movieIdBatch, genreBatch, labelBatch]);

                } catch (error) {
                    console.error(`Error in batch ${i/batchSize}:`, error);
                    break;
                }
            }

            if (batchCount > 0) {
                const avgLoss = (await epochLoss.data())[0] / batchCount;
                const avgAcc = (await epochAcc.data())[0] / batchCount;
                
                history.loss.push(avgLoss);
                history.acc.push(avgAcc);

                console.log(`Epoch ${epoch + 1}/${epochs} - loss: ${avgLoss.toFixed(4)} - acc: ${avgAcc.toFixed(4)}`);
            }

            epochLoss.dispose();
            epochAcc.dispose();
        }

        // Cleanup
        movieIds.dispose();
        genres.dispose();

        return history;
    }

    async getUserEmbedding(userId) {
        if (!this.userTower) {
            throw new Error('User tower not initialized');
        }
        
        const userInput = tf.tensor1d([userId], 'int32');
        const embedding = this.userTower.predict(userInput);
        userInput.dispose();
        return embedding;
    }

    async getItemEmbedding(movieId, genres) {
        if (!this.itemTower) {
            throw new Error('Item tower not initialized');
        }
        
        const movieIdTensor = tf.tensor1d([movieId], 'int32');
        const genreTensor = tf.tensor2d([genres]);
        
        const embedding = this.itemTower.predict([movieIdTensor, genreTensor]);
        
        movieIdTensor.dispose();
        genreTensor.dispose();
        
        return embedding;
    }

    compileModel() {
        this.model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });
    }
}

class BaselineTwoTower extends BaseTwoTower {
    buildModel() {
        console.log('Building Baseline Matrix Factorization model...');
        
        // User tower - simple embedding
        const userInput = tf.input({ shape: [1], name: 'user_input', dtype: 'int32' });
        const userEmbedding = tf.layers.embedding({
            inputDim: this.numUsers,
            outputDim: this.embeddingDim,
            name: 'user_embedding'
        }).apply(userInput);
        const userFlatten = tf.layers.flatten().apply(userEmbedding);
        this.userTower = tf.model({ inputs: userInput, outputs: userFlatten });

        // Item tower - simple embedding + genre projection
        const movieIdInput = tf.input({ shape: [1], name: 'movie_id_input', dtype: 'int32' });
        const genreInput = tf.input({ shape: [this.numGenres], name: 'genre_input' });
        
        // Movie embedding
        const movieEmbedding = tf.layers.embedding({
            inputDim: this.numItems,
            outputDim: this.embeddingDim,
            name: 'movie_embedding'
        }).apply(movieIdInput);
        const movieFlatten = tf.layers.flatten().apply(movieEmbedding);
        
        // Genre projection to match embedding dim
        const genreProjection = tf.layers.dense({
            units: this.embeddingDim,
            activation: 'linear',
            name: 'genre_projection'
        }).apply(genreInput);
        
        // Combine movie embedding and genre features
        const combined = tf.layers.add().apply([movieFlatten, genreProjection]);
        
        this.itemTower = tf.model({ 
            inputs: [movieIdInput, genreInput], 
            outputs: combined 
        });

        // Combined model for training
        const userOutput = this.userTower.apply(userInput);
        const itemOutput = this.itemTower.apply([movieIdInput, genreInput]);
        
        // Dot product
        const dotProduct = tf.layers.dot({ axes: 1 }).apply([userOutput, itemOutput]);
        
        // Prediction
        const prediction = tf.layers.dense({
            units: 1,
            activation: 'sigmoid',
            name: 'prediction'
        }).apply(dotProduct);

        this.model = tf.model({
            inputs: [userInput, movieIdInput, genreInput],
            outputs: prediction
        });

        this.compileModel();
        console.log('Baseline model built successfully');
        this.model.summary();
    }
}

class DeepLearningTwoTower extends BaseTwoTower {
    buildModel() {
        console.log('Building Deep Learning Two-Tower model...');
        
        // User tower
        const userInput = tf.input({ shape: [1], name: 'user_input', dtype: 'int32' });
        const userEmbedding = tf.layers.embedding({
            inputDim: this.numUsers,
            outputDim: 64,
            name: 'user_embedding'
        }).apply(userInput);
        const userFlatten = tf.layers.flatten().apply(userEmbedding);
        
        const userDense1 = tf.layers.dense({
            units: 32,
            activation: 'relu'
        }).apply(userFlatten);
        
        const userOutput = tf.layers.dense({
            units: this.embeddingDim,
            activation: 'linear'
        }).apply(userDense1);
        
        this.userTower = tf.model({ inputs: userInput, outputs: userOutput });

        // Item tower
        const movieIdInput = tf.input({ shape: [1], name: 'movie_id_input', dtype: 'int32' });
        const genreInput = tf.input({ shape: [this.numGenres], name: 'genre_input' });
        
        const movieEmbedding = tf.layers.embedding({
            inputDim: this.numItems,
            outputDim: 32,
            name: 'movie_embedding'
        }).apply(movieIdInput);
        const movieFlatten = tf.layers.flatten().apply(movieEmbedding);
        
        // Process genres separately
        const genreDense = tf.layers.dense({
            units: 16,
            activation: 'relu'
        }).apply(genreInput);
        
        // Combine
        const combined = tf.layers.concatenate().apply([movieFlatten, genreDense]);
        
        const itemDense1 = tf.layers.dense({
            units: 32,
            activation: 'relu'
        }).apply(combined);
        
        const itemOutput = tf.layers.dense({
            units: this.embeddingDim,
            activation: 'linear'
        }).apply(itemDense1);
        
        this.itemTower = tf.model({ 
            inputs: [movieIdInput, genreInput], 
            outputs: itemOutput 
        });

        // Combined model
        const userOutputFinal = this.userTower.apply(userInput);
        const itemOutputFinal = this.itemTower.apply([movieIdInput, genreInput]);
        
        const dotProduct = tf.layers.dot({ axes: 1 }).apply([userOutputFinal, itemOutputFinal]);
        
        const prediction = tf.layers.dense({
            units: 1,
            activation: 'sigmoid'
        }).apply(dotProduct);

        this.model = tf.model({
            inputs: [userInput, movieIdInput, genreInput],
            outputs: prediction
        });

        this.compileModel();
        console.log('Deep Learning model built successfully');
        this.model.summary();
    }
}

class MLPTwoTower extends BaseTwoTower {
    buildModel() {
        console.log('Building MLP Two-Tower model...');
        
        // User tower
        const userInput = tf.input({ shape: [1], name: 'user_input', dtype: 'int32' });
        const userEmbedding = tf.layers.embedding({
            inputDim: this.numUsers,
            outputDim: 32,
            name: 'user_embedding'
        }).apply(userInput);
        const userFlatten = tf.layers.flatten().apply(userEmbedding);
        
        const userHidden1 = tf.layers.dense({
            units: 16,
            activation: 'relu'
        }).apply(userFlatten);
        
        const userOutput = tf.layers.dense({
            units: this.embeddingDim,
            activation: 'linear'
        }).apply(userHidden1);
        
        this.userTower = tf.model({ inputs: userInput, outputs: userOutput });

        // Item tower
        const movieIdInput = tf.input({ shape: [1], name: 'movie_id_input', dtype: 'int32' });
        const genreInput = tf.input({ shape: [this.numGenres], name: 'genre_input' });
        
        const movieEmbedding = tf.layers.embedding({
            inputDim: this.numItems,
            outputDim: 32,
            name: 'movie_embedding'
        }).apply(movieIdInput);
        const movieFlatten = tf.layers.flatten().apply(movieEmbedding);
        
        // Process genres
        const genreDense = tf.layers.dense({
            units: 8,
            activation: 'relu'
        }).apply(genreInput);
        
        // Combine
        const combined = tf.layers.concatenate().apply([movieFlatten, genreDense]);
        
        const itemHidden1 = tf.layers.dense({
            units: 24,
            activation: 'relu'
        }).apply(combined);
        
        const itemOutput = tf.layers.dense({
            units: this.embeddingDim,
            activation: 'linear'
        }).apply(itemHidden1);
        
        this.itemTower = tf.model({ 
            inputs: [movieIdInput, genreInput], 
            outputs: itemOutput 
        });

        // Combined model
        const userOutputFinal = this.userTower.apply(userInput);
        const itemOutputFinal = this.itemTower.apply([movieIdInput, genreInput]);
        
        const dotProduct = tf.layers.dot({ axes: 1 }).apply([userOutputFinal, itemOutputFinal]);
        
        const prediction = tf.layers.dense({
            units: 1,
            activation: 'sigmoid'
        }).apply(dotProduct);

        this.model = tf.model({
            inputs: [userInput, movieIdInput, genreInput],
            outputs: prediction
        });

        this.compileModel();
        console.log('MLP model built successfully');
        this.model.summary();
    }
}
