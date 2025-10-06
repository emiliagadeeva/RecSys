class BaseTwoTower {
    constructor(embeddingDim, numUsers, numItems, numGenres) {
        this.embeddingDim = embeddingDim;
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.numGenres = numGenres;
        this.model = null;
        this.userTower = null;
        this.itemTower = null;
        this.updateTrainingStatus = null;
    }

    buildModel() {
        throw new Error('buildModel must be implemented by subclass');
    }

    async train(userInput, movieInput, ratings, epochs = 5, batchSize = 32) {
        if (!this.model) {
            this.buildModel();
        }

        console.log(`Starting training for ${this.constructor.name}`);
        const history = {
            loss: [],
            acc: []
        };

        const numSamples = userInput.shape[0];
        console.log(`Training on ${numSamples} samples`);

        for (let epoch = 0; epoch < epochs; epoch++) {
            let epochLoss = 0;
            let epochAcc = 0;
            let batchCount = 0;

            // Create shuffled indices for this epoch
            const indices = tf.util.createShuffledIndices(numSamples);
            
            for (let i = 0; i < numSamples; i += batchSize) {
                const end = Math.min(i + batchSize, numSamples);
                
                // Get batch using shuffled indices
                const batchIndices = indices.slice(i, end);
                const userBatch = userInput.gather(batchIndices);
                
                // Prepare movie data for this batch
                const movieIdBatchValues = [];
                const genreBatchValues = [];
                
                for (let j = i; j < end; j++) {
                    const originalIndex = batchIndices[j - i];
                    movieIdBatchValues.push(movieInput[originalIndex].movieId);
                    genreBatchValues.push(movieInput[originalIndex].genres);
                }
                
                const movieIdBatch = tf.tensor1d(movieIdBatchValues, 'int32');
                const genreBatch = tf.tensor2d(genreBatchValues);
                const labelBatch = ratings.gather(batchIndices);

                try {
                    const result = await this.model.trainOnBatch(
                        [userBatch, movieIdBatch, genreBatch], 
                        labelBatch
                    );

                    epochLoss += result[0];
                    epochAcc += result[1];
                    batchCount++;

                } catch (error) {
                    console.error('Error in training batch:', error);
                } finally {
                    // Cleanup
                    tf.dispose([userBatch, movieIdBatch, genreBatch, labelBatch]);
                }
            }

            if (batchCount > 0) {
                const avgLoss = epochLoss / batchCount;
                const avgAcc = epochAcc / batchCount;
                
                history.loss.push(avgLoss);
                history.acc.push(avgAcc);

                console.log(`Epoch ${epoch + 1}/${epochs} - loss: ${avgLoss.toFixed(4)} - acc: ${avgAcc.toFixed(4)}`);
                
                if (this.updateTrainingStatus) {
                    this.updateTrainingStatus(`Epoch ${epoch + 1}/${epochs} - Loss: ${avgLoss.toFixed(4)}`);
                }
            }
        }

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

    setStatusCallback(callback) {
        this.updateTrainingStatus = callback;
    }
}

class WithoutDLTwoTower extends BaseTwoTower {
    buildModel() {
        console.log('Building Without DL Two-Tower model...');
        
        // User Tower - Simple embedding
        const userInput = tf.input({ shape: [1], name: 'user_input', dtype: 'int32' });
        const userEmbedding = tf.layers.embedding({
            inputDim: this.numUsers,
            outputDim: this.embeddingDim,
            name: 'user_embedding'
        }).apply(userInput);
        const userFlatten = tf.layers.flatten().apply(userEmbedding);
        this.userTower = tf.model({ inputs: userInput, outputs: userFlatten });

        // Item Tower - Embedding + Genre features
        const movieIdInput = tf.input({ shape: [1], name: 'movie_id_input', dtype: 'int32' });
        const genreInput = tf.input({ shape: [this.numGenres], name: 'genre_input' });
        
        const movieEmbedding = tf.layers.embedding({
            inputDim: this.numItems,
            outputDim: this.embeddingDim,
            name: 'movie_embedding'
        }).apply(movieIdInput);
        const movieFlatten = tf.layers.flatten().apply(movieEmbedding);
        
        // Project genre features to embedding dimension
        const genreProjection = tf.layers.dense({
            units: this.embeddingDim,
            activation: 'linear',
            name: 'genre_projection'
        }).apply(genreInput);
        
        // Combine movie embedding and genre features
        const itemOutput = tf.layers.add().apply([movieFlatten, genreProjection]);
        this.itemTower = tf.model({ 
            inputs: [movieIdInput, genreInput], 
            outputs: itemOutput 
        });

        // Combined model
        const userOutput = this.userTower.apply(userInput);
        const itemOutputFinal = this.itemTower.apply([movieIdInput, genreInput]);
        
        const dotProduct = tf.layers.dot({ axes: 1 }).apply([userOutput, itemOutputFinal]);
        
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
        console.log('Without DL model built successfully');
    }
}

class MLPTwoTower extends BaseTwoTower {
    buildModel() {
        console.log('Building MLP Two-Tower model...');
        
        // User Tower with MLP
        const userInput = tf.input({ shape: [1], name: 'user_input', dtype: 'int32' });
        const userEmbedding = tf.layers.embedding({
            inputDim: this.numUsers,
            outputDim: 32,
            name: 'user_embedding'
        }).apply(userInput);
        const userFlatten = tf.layers.flatten().apply(userEmbedding);
        
        // MLP with hidden layer (ReLU activation)
        const userHidden = tf.layers.dense({
            units: 16,
            activation: 'relu',
            name: 'user_hidden'
        }).apply(userFlatten);
        
        const userOutput = tf.layers.dense({
            units: this.embeddingDim,
            activation: 'linear',
            name: 'user_output'
        }).apply(userHidden);
        
        this.userTower = tf.model({ inputs: userInput, outputs: userOutput });

        // Item Tower with MLP
        const movieIdInput = tf.input({ shape: [1], name: 'movie_id_input', dtype: 'int32' });
        const genreInput = tf.input({ shape: [this.numGenres], name: 'genre_input' });
        
        const movieEmbedding = tf.layers.embedding({
            inputDim: this.numItems,
            outputDim: 32,
            name: 'movie_embedding'
        }).apply(movieIdInput);
        const movieFlatten = tf.layers.flatten().apply(movieEmbedding);
        
        // Process genre features with MLP
        const genreHidden = tf.layers.dense({
            units: 8,
            activation: 'relu',
            name: 'genre_hidden'
        }).apply(genreInput);
        
        // Combine movie and genre features
        const combined = tf.layers.concatenate().apply([movieFlatten, genreHidden]);
        
        // MLP for combined features
        const itemHidden = tf.layers.dense({
            units: 16,
            activation: 'relu',
            name: 'item_hidden'
        }).apply(combined);
        
        const itemOutput = tf.layers.dense({
            units: this.embeddingDim,
            activation: 'linear',
            name: 'item_output'
        }).apply(itemHidden);
        
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
            activation: 'sigmoid',
            name: 'prediction'
        }).apply(dotProduct);

        this.model = tf.model({
            inputs: [userInput, movieIdInput, genreInput],
            outputs: prediction
        });

        this.compileModel();
        console.log('MLP Two-Tower model built successfully');
    }
}

class DeepLearningTwoTower extends BaseTwoTower {
    buildModel() {
        console.log('Building Deep Learning Two-Tower model...');
        
        // User Tower - Deep architecture
        const userInput = tf.input({ shape: [1], name: 'user_input', dtype: 'int32' });
        const userEmbedding = tf.layers.embedding({
            inputDim: this.numUsers,
            outputDim: 64,
            name: 'user_embedding'
        }).apply(userInput);
        const userFlatten = tf.layers.flatten().apply(userEmbedding);
        
        // Multiple hidden layers
        const userDense1 = tf.layers.dense({
            units: 32,
            activation: 'relu',
            name: 'user_dense1'
        }).apply(userFlatten);
        
        const userDense2 = tf.layers.dense({
            units: 16,
            activation: 'relu',
            name: 'user_dense2'
        }).apply(userDense1);
        
        const userOutput = tf.layers.dense({
            units: this.embeddingDim,
            activation: 'tanh',
            name: 'user_output'
        }).apply(userDense2);
        
        this.userTower = tf.model({ inputs: userInput, outputs: userOutput });

        // Item Tower - Deep architecture with genre integration
        const movieIdInput = tf.input({ shape: [1], name: 'movie_id_input', dtype: 'int32' });
        const genreInput = tf.input({ shape: [this.numGenres], name: 'genre_input' });
        
        const movieEmbedding = tf.layers.embedding({
            inputDim: this.numItems,
            outputDim: 64,
            name: 'movie_embedding'
        }).apply(movieIdInput);
        const movieFlatten = tf.layers.flatten().apply(movieEmbedding);
        
        // Deep processing for genres
        const genreDense1 = tf.layers.dense({
            units: 32,
            activation: 'relu',
            name: 'genre_dense1'
        }).apply(genreInput);
        
        const genreDense2 = tf.layers.dense({
            units: 16,
            activation: 'relu',
            name: 'genre_dense2'
        }).apply(genreDense1);
        
        // Combine and further process
        const combined = tf.layers.concatenate().apply([movieFlatten, genreDense2]);
        
        const itemDense1 = tf.layers.dense({
            units: 32,
            activation: 'relu',
            name: 'item_dense1'
        }).apply(combined);
        
        const itemDense2 = tf.layers.dense({
            units: 16,
            activation: 'relu',
            name: 'item_dense2'
        }).apply(itemDense1);
        
        const itemOutput = tf.layers.dense({
            units: this.embeddingDim,
            activation: 'tanh',
            name: 'item_output'
        }).apply(itemDense2);
        
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
            activation: 'sigmoid',
            name: 'prediction'
        }).apply(dotProduct);

        this.model = tf.model({
            inputs: [userInput, movieIdInput, genreInput],
            outputs: prediction
        });

        this.compileModel();
        console.log('Deep Learning Two-Tower model built successfully');
    }
}
