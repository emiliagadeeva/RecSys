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

    async train(userInput, movieInput, ratings, epochs = 5, batchSize = 256) {
        if (!this.model) {
            this.buildModel();
        }

        console.log(`Starting training for ${this.constructor.name}`);
        const history = {
            loss: [],
            acc: []
        };

        const dataset = this.createDataset(userInput, movieInput, ratings, batchSize);
        const numBatches = Math.ceil(userInput.shape[0] / batchSize);

        for (let epoch = 0; epoch < epochs; epoch++) {
            let epochLoss = 0;
            let epochAcc = 0;
            let batchCount = 0;

            const iterator = dataset.iterator();
            
            for (let batch = 0; batch < numBatches; batch++) {
                const batchResult = await iterator.next();
                if (batchResult.done) break;

                const { userBatch, movieIdBatch, genreBatch, labelBatch } = batchResult.value;
                
                try {
                    const result = await this.model.trainOnBatch(
                        [userBatch, movieIdBatch, genreBatch], 
                        labelBatch
                    );

                    epochLoss += result[0];
                    epochAcc += result[1];
                    batchCount++;

                } catch (error) {
                    console.error(`Error in batch ${batch}:`, error);
                } finally {
                    // Clean up tensors
                    tf.dispose([userBatch, movieIdBatch, genreBatch, labelBatch]);
                }
            }

            if (batchCount > 0) {
                const avgLoss = epochLoss / batchCount;
                const avgAcc = epochAcc / batchCount;
                
                history.loss.push(avgLoss);
                history.acc.push(avgAcc);

                console.log(`Epoch ${epoch + 1}/${epochs} - loss: ${avgLoss.toFixed(4)} - acc: ${avgAcc.toFixed(4)}`);
            }
        }

        return history;
    }

    createDataset(userInput, movieInput, ratings, batchSize) {
        const dataSize = userInput.shape[0];
        
        return {
            iterator: () => {
                let position = 0;
                
                return {
                    next: async () => {
                        if (position >= dataSize) {
                            return { done: true };
                        }

                        const end = Math.min(position + batchSize, dataSize);
                        
                        const userBatch = userInput.slice([position], [end - position]);
                        const movieIdValues = [];
                        const genreValues = [];
                        
                        for (let i = position; i < end; i++) {
                            movieIdValues.push(movieInput[i].movieId);
                            genreValues.push(movieInput[i].genres);
                        }
                        
                        const movieIdBatch = tf.tensor1d(movieIdValues, 'int32');
                        const genreBatch = tf.tensor2d(genreValues);
                        const labelBatch = ratings.slice([position], [end - position]);

                        position = end;

                        return {
                            value: { userBatch, movieIdBatch, genreBatch, labelBatch },
                            done: false
                        };
                    }
                };
            }
        };
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
        
        const finalEmbeddingDim = this.embeddingDim;
        
        // User tower - simple embedding
        const userInput = tf.input({ shape: [1], name: 'user_input', dtype: 'int32' });
        const userEmbedding = tf.layers.embedding({
            inputDim: this.numUsers,
            outputDim: finalEmbeddingDim,
            name: 'user_embedding'
        }).apply(userInput);
        const userFlatten = tf.layers.flatten().apply(userEmbedding);
        this.userTower = tf.model({ inputs: userInput, outputs: userFlatten });

        // Item tower
        const movieIdInput = tf.input({ shape: [1], name: 'movie_id_input', dtype: 'int32' });
        const genreInput = tf.input({ shape: [this.numGenres], name: 'genre_input' });
        
        const movieEmbedding = tf.layers.embedding({
            inputDim: this.numItems,
            outputDim: finalEmbeddingDim - this.numGenres, // Reserve space for genres
            name: 'movie_embedding'
        }).apply(movieIdInput);
        const movieFlatten = tf.layers.flatten().apply(movieEmbedding);
        
        // Concatenate movie embedding with genre features
        const itemConcat = tf.layers.concatenate().apply([movieFlatten, genreInput]);
        
        // Project to final embedding dimension
        const itemProjection = tf.layers.dense({
            units: finalEmbeddingDim,
            activation: 'linear',
            name: 'item_projection'
        }).apply(itemConcat);
        
        this.itemTower = tf.model({ 
            inputs: [movieIdInput, genreInput], 
            outputs: itemProjection 
        });

        // Combined model
        const userOutput = this.userTower.apply(userInput);
        const itemOutput = this.itemTower.apply([movieIdInput, genreInput]);
        
        // Dot product between user and item embeddings
        const dotProduct = tf.layers.dot({ axes: -1, normalize: false }).apply([
            userOutput, itemOutput
        ]);
        
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
    }
}

class DeepLearningTwoTower extends BaseTwoTower {
    buildModel() {
        console.log('Building Deep Learning Two-Tower model...');
        
        const finalEmbeddingDim = this.embeddingDim;
        
        // User tower - deeper architecture
        const userInput = tf.input({ shape: [1], name: 'user_input', dtype: 'int32' });
        const userEmbedding = tf.layers.embedding({
            inputDim: this.numUsers,
            outputDim: 64,
            name: 'user_embedding'
        }).apply(userInput);
        const userFlatten = tf.layers.flatten().apply(userEmbedding);
        
        // Deep layers for user tower
        const userDense1 = tf.layers.dense({
            units: 32,
            activation: 'relu',
            name: 'user_dense_1'
        }).apply(userFlatten);
        
        const userOutput = tf.layers.dense({
            units: finalEmbeddingDim,
            activation: 'linear',
            name: 'user_output'
        }).apply(userDense1);
        
        this.userTower = tf.model({ inputs: userInput, outputs: userOutput });

        // Item tower - deeper architecture
        const movieIdInput = tf.input({ shape: [1], name: 'movie_id_input', dtype: 'int32' });
        const genreInput = tf.input({ shape: [this.numGenres], name: 'genre_input' });
        
        const movieEmbedding = tf.layers.embedding({
            inputDim: this.numItems,
            outputDim: 32,
            name: 'movie_embedding'
        }).apply(movieIdInput);
        const movieFlatten = tf.layers.flatten().apply(movieEmbedding);
        
        // Concatenate with genre features
        const itemConcat = tf.layers.concatenate().apply([movieFlatten, genreInput]);
        
        // Deep layers for item tower
        const itemDense1 = tf.layers.dense({
            units: 64,
            activation: 'relu',
            name: 'item_dense_1'
        }).apply(itemConcat);
        
        const itemOutput = tf.layers.dense({
            units: finalEmbeddingDim,
            activation: 'linear',
            name: 'item_output'
        }).apply(itemDense1);
        
        this.itemTower = tf.model({ 
            inputs: [movieIdInput, genreInput], 
            outputs: itemOutput 
        });

        // Combined model
        const userOutputFinal = this.userTower.apply(userInput);
        const itemOutputFinal = this.itemTower.apply([movieIdInput, genreInput]);
        
        // Dot product between user and item embeddings
        const dotProduct = tf.layers.dot({ axes: -1, normalize: false }).apply([
            userOutputFinal, itemOutputFinal
        ]);
        
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
        console.log('Deep Learning model built successfully');
    }
}

class MLPTwoTower extends BaseTwoTower {
    buildModel() {
        console.log('Building MLP Two-Tower model...');
        
        const finalEmbeddingDim = this.embeddingDim;
        
        // User tower with MLP
        const userInput = tf.input({ shape: [1], name: 'user_input', dtype: 'int32' });
        const userEmbedding = tf.layers.embedding({
            inputDim: this.numUsers,
            outputDim: 32,
            name: 'user_embedding'
        }).apply(userInput);
        const userFlatten = tf.layers.flatten().apply(userEmbedding);
        
        // MLP with hidden layers
        const userHidden1 = tf.layers.dense({
            units: 16,
            activation: 'relu',
            name: 'user_mlp_1'
        }).apply(userFlatten);
        
        const userOutput = tf.layers.dense({
            units: finalEmbeddingDim,
            activation: 'linear',
            name: 'user_mlp_output'
        }).apply(userHidden1);
        
        this.userTower = tf.model({ inputs: userInput, outputs: userOutput });

        // Item tower with MLP
        const movieIdInput = tf.input({ shape: [1], name: 'movie_id_input', dtype: 'int32' });
        const genreInput = tf.input({ shape: [this.numGenres], name: 'genre_input' });
        
        const movieEmbedding = tf.layers.embedding({
            inputDim: this.numItems,
            outputDim: 32,
            name: 'movie_embedding'
        }).apply(movieIdInput);
        const movieFlatten = tf.layers.flatten().apply(movieEmbedding);
        
        // Concatenate with genre features
        const itemConcat = tf.layers.concatenate().apply([movieFlatten, genreInput]);
        
        // MLP with hidden layers
        const itemHidden1 = tf.layers.dense({
            units: 32,
            activation: 'relu',
            name: 'item_mlp_1'
        }).apply(itemConcat);
        
        const itemOutput = tf.layers.dense({
            units: finalEmbeddingDim,
            activation: 'linear',
            name: 'item_mlp_output'
        }).apply(itemHidden1);
        
        this.itemTower = tf.model({ 
            inputs: [movieIdInput, genreInput], 
            outputs: itemOutput 
        });

        // Combined model
        const userOutputFinal = this.userTower.apply(userInput);
        const itemOutputFinal = this.itemTower.apply([movieIdInput, genreInput]);
        
        // Dot product between user and item embeddings
        const dotProduct = tf.layers.dot({ axes: -1, normalize: false }).apply([
            userOutputFinal, itemOutputFinal
        ]);
        
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
        console.log('MLP model built successfully');
    }
}
