class TwoTowerBase {
    constructor(numUsers, numItems, embeddingDim, numGenres) {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.embeddingDim = embeddingDim;
        this.numGenres = numGenres;
        this.model = null;
        this.isTraining = false;
    }

    createModel() {
        throw new Error('createModel must be implemented by subclass');
    }

    async train(ratings, movies, epochs = 3, batchSize = 32) {
        if (this.isTraining) {
            throw new Error('Model is already training');
        }

        this.isTraining = true;

        try {
            if (!this.model) {
                this.model = this.createModel();
                console.log('Model created successfully');
            }

            // Prepare training data
            const trainingData = this.prepareTrainingData(ratings, movies);
            if (!trainingData) {
                throw new Error('Failed to prepare training data');
            }

            const { userInput, itemInput, genreFeatures, labels } = trainingData;
            const history = { loss: [] };

            const optimizer = tf.train.adam(0.01); // Increased learning rate

            for (let epoch = 0; epoch < epochs; epoch++) {
                console.log(`Starting epoch ${epoch + 1}`);
                const epochLoss = await this.trainEpoch(optimizer, userInput, itemInput, genreFeatures, labels, batchSize);
                history.loss.push(epochLoss);
                
                console.log(`Epoch ${epoch + 1}/${epochs}, Loss: ${epochLoss.toFixed(4)}`);
                
                // Update UI periodically
                await tf.nextFrame();
            }

            console.log('Training completed successfully');
            return history;

        } catch (error) {
            console.error('Training error:', error);
            throw error;
        } finally {
            this.isTraining = false;
        }
    }

    prepareTrainingData(ratings, movies) {
        if (!ratings || ratings.length === 0) {
            console.error('No ratings provided');
            return null;
        }

        const userInput = [];
        const itemInput = [];
        const labels = [];
        const genreFeaturesArray = [];

        // Use only a subset for faster training
        const trainingRatings = ratings.slice(0, 5000);

        for (const rating of trainingRatings) {
            const userId = rating.user_id - 1;
            const itemId = rating.item_id - 1;

            if (userId >= 0 && userId < this.numUsers && itemId >= 0 && itemId < this.numItems) {
                userInput.push(userId);
                itemInput.push(itemId);
                labels.push(rating.rating >= 4 ? 1 : 0);

                // Get genre features
                const movie = movies[rating.item_id];
                if (movie && movie.genreFeatures) {
                    genreFeaturesArray.push(movie.genreFeatures);
                } else {
                    genreFeaturesArray.push(Array(this.numGenres).fill(0));
                }
            }
        }

        if (userInput.length === 0) {
            console.error('No valid training examples found');
            return null;
        }

        console.log(`Prepared ${userInput.length} training examples`);

        return {
            userInput: tf.tensor1d(userInput, 'int32'),
            itemInput: tf.tensor1d(itemInput, 'int32'),
            genreFeatures: tf.tensor2d(genreFeaturesArray, [userInput.length, this.numGenres], 'float32'),
            labels: tf.tensor1d(labels, 'float32')
        };
    }

    async trainEpoch(optimizer, userInput, itemInput, genreFeatures, labels, batchSize) {
        const userArray = await userInput.array();
        const itemArray = await itemInput.array();
        const genreArray = await genreFeatures.array();
        const labelArray = await labels.array();

        const numBatches = Math.ceil(userArray.length / batchSize);
        let totalLoss = 0;
        let processedBatches = 0;

        for (let i = 0; i < numBatches; i++) {
            const start = i * batchSize;
            const end = Math.min(start + batchSize, userArray.length);
            
            const batchUser = tf.tensor1d(userArray.slice(start, end), 'int32');
            const batchItem = tf.tensor1d(itemArray.slice(start, end), 'int32');
            const batchGenre = tf.tensor2d(genreArray.slice(start, end), [end - start, this.numGenres], 'float32');
            const batchLabel = tf.tensor1d(labelArray.slice(start, end), 'float32');

            const loss = this.trainStep(optimizer, batchUser, batchItem, batchGenre, batchLabel);
            totalLoss += loss;
            processedBatches++;

            tf.dispose([batchUser, batchItem, batchGenre, batchLabel]);
        }

        // Clean up main tensors
        tf.dispose([userInput, itemInput, genreFeatures, labels]);

        return processedBatches > 0 ? totalLoss / processedBatches : 0;
    }

    trainStep(optimizer, userIds, itemIds, genreFeatures, labels) {
        return tf.tidy(() => {
            const lossValue = optimizer.minimize(() => {
                // Get embeddings from both towers
                const userEmbedding = this.model.layers[0].apply(userIds);
                const itemEmbedding = this.model.layers[1].apply([itemIds, genreFeatures]);
                
                // Calculate dot product scores
                const scores = tf.sum(tf.mul(userEmbedding, itemEmbedding), 1);
                const predictions = tf.sigmoid(scores);
                
                // Binary cross entropy loss
                return tf.losses.sigmoidCrossEntropy(labels, predictions);
            }, true);

            return lossValue ? lossValue.dataSync()[0] : 0;
        });
    }

    async recommend(userId, movies, topK = 10) {
        if (!this.model) {
            throw new Error('Model not trained. Please train the model first.');
        }

        if (userId < 1 || userId > this.numUsers) {
            console.warn(`User ID ${userId} is out of range. Using user 1.`);
            userId = 1;
        }

        return tf.tidy(() => {
            try {
                const userTensor = tf.tensor1d([userId - 1], 'int32');
                const allItemIds = Array.from({ length: this.numItems }, (_, i) => i);
                const itemTensor = tf.tensor1d(allItemIds, 'int32');
                
                // Prepare genre features for all items
                const allGenreFeatures = [];
                for (let i = 1; i <= this.numItems; i++) {
                    if (movies[i] && movies[i].genreFeatures) {
                        allGenreFeatures.push(movies[i].genreFeatures);
                    } else {
                        allGenreFeatures.push(Array(this.numGenres).fill(0));
                    }
                }
                const genreTensor = tf.tensor2d(allGenreFeatures, [this.numItems, this.numGenres], 'float32');

                // Get embeddings
                const userEmbedding = this.model.layers[0].apply(userTensor);
                const itemEmbeddings = this.model.layers[1].apply([itemTensor, genreTensor]);
                
                // Calculate scores - ensure shapes match
                const userEmbeddingRepeated = userEmbedding.tile([this.numItems, 1]);
                
                // Dot product between user and item embeddings
                const scores = tf.sum(tf.mul(userEmbeddingRepeated, itemEmbeddings), 1);
                
                // Get top K recommendations
                const actualTopK = Math.min(topK, this.numItems);
                const { values, indices } = tf.topk(scores, actualTopK);
                
                const topScores = values.arraySync();
                const topIndices = indices.arraySync();
                
                const recommendations = [];
                for (let i = 0; i < actualTopK; i++) {
                    const itemId = topIndices[i] + 1; // Convert back to 1-based
                    const movie = movies[itemId];
                    if (movie) {
                        recommendations.push({
                            id: itemId,
                            title: movie.title,
                            score: topScores[i],
                            genres: movie.genres || ['Unknown']
                        });
                    } else {
                        recommendations.push({
                            id: itemId,
                            title: `Movie ${itemId}`,
                            score: topScores[i],
                            genres: ['Unknown']
                        });
                    }
                }
                
                return recommendations;
            } catch (error) {
                console.error('Error in recommend method:', error);
                return [];
            }
        });
    }
}

class WithoutDLTwoTower extends TwoTowerBase {
    createModel() {
        try {
            // User tower: simple embedding
            const userInput = tf.input({ shape: [1], dtype: 'int32', name: 'user_input' });
            const userEmbedding = tf.layers.embedding({
                inputDim: this.numUsers,
                outputDim: this.embeddingDim,
                embeddingsInitializer: 'glorotNormal',
                name: 'user_embedding'
            }).apply(userInput);
            const userOutput = tf.layers.flatten().apply(userEmbedding);
            
            // Item tower: embedding + genre features
            const itemInput = tf.input({ shape: [1], dtype: 'int32', name: 'item_input' });
            const genreInput = tf.input({ shape: [this.numGenres], dtype: 'float32', name: 'genre_input' });
            
            const itemEmbedding = tf.layers.embedding({
                inputDim: this.numItems,
                outputDim: this.embeddingDim,
                embeddingsInitializer: 'glorotNormal',
                name: 'item_embedding'
            }).apply(itemInput);
            const itemFlatten = tf.layers.flatten().apply(itemEmbedding);
            
            // Combine item embedding with genre features
            const itemConcat = tf.layers.concatenate().apply([itemFlatten, genreInput]);
            const itemOutput = tf.layers.dense({
                units: this.embeddingDim,
                activation: 'linear',
                kernelInitializer: 'glorotNormal',
                name: 'item_projection'
            }).apply(itemConcat);
            
            // Create tower models
            const userTower = tf.model({ 
                inputs: userInput, 
                outputs: userOutput,
                name: 'user_tower'
            });
            
            const itemTower = tf.model({ 
                inputs: [itemInput, genreInput], 
                outputs: itemOutput,
                name: 'item_tower'
            });
            
            // Create combined model for training
            const combinedUserInput = tf.input({ shape: [1], dtype: 'int32', name: 'combined_user_input' });
            const combinedItemInput = tf.input({ shape: [1], dtype: 'int32', name: 'combined_item_input' });
            const combinedGenreInput = tf.input({ shape: [this.numGenres], dtype: 'float32', name: 'combined_genre_input' });
            
            const userTowerOutput = userTower.apply(combinedUserInput);
            const itemTowerOutput = itemTower.apply([combinedItemInput, combinedGenreInput]);
            
            // Dot product similarity
            const dotProduct = tf.layers.dot({ 
                axes: 1, 
                normalize: false 
            }).apply([userTowerOutput, itemTowerOutput]);
            
            const combinedModel = tf.model({
                inputs: [combinedUserInput, combinedItemInput, combinedGenreInput],
                outputs: dotProduct,
                name: 'combined_model'
            });
            
            console.log('WithoutDLTwoTower model created successfully');
            return combinedModel;
            
        } catch (error) {
            console.error('Error creating WithoutDLTwoTower model:', error);
            throw error;
        }
    }
}

class MLPTwoTower extends TwoTowerBase {
    createModel() {
        try {
            // User tower: embedding + MLP
            const userInput = tf.input({ shape: [1], dtype: 'int32', name: 'user_input' });
            const userEmbedding = tf.layers.embedding({
                inputDim: this.numUsers,
                outputDim: this.embeddingDim,
                embeddingsInitializer: 'glorotNormal',
                name: 'user_embedding'
            }).apply(userInput);
            const userFlatten = tf.layers.flatten().apply(userEmbedding);
            
            // MLP for user tower
            const userHidden = tf.layers.dense({
                units: 64,
                activation: 'relu',
                kernelInitializer: 'heNormal',
                name: 'user_hidden'
            }).apply(userFlatten);
            
            const userOutput = tf.layers.dense({
                units: this.embeddingDim,
                activation: 'linear',
                kernelInitializer: 'glorotNormal',
                name: 'user_output'
            }).apply(userHidden);
            
            // Item tower: embedding + genre features + MLP
            const itemInput = tf.input({ shape: [1], dtype: 'int32', name: 'item_input' });
            const genreInput = tf.input({ shape: [this.numGenres], dtype: 'float32', name: 'genre_input' });
            
            const itemEmbedding = tf.layers.embedding({
                inputDim: this.numItems,
                outputDim: this.embeddingDim,
                embeddingsInitializer: 'glorotNormal',
                name: 'item_embedding'
            }).apply(itemInput);
            const itemFlatten = tf.layers.flatten().apply(itemEmbedding);
            
            // Combine and process with MLP
            const itemConcat = tf.layers.concatenate().apply([itemFlatten, genreInput]);
            const itemHidden = tf.layers.dense({
                units: 64,
                activation: 'relu',
                kernelInitializer: 'heNormal',
                name: 'item_hidden'
            }).apply(itemConcat);
            
            const itemOutput = tf.layers.dense({
                units: this.embeddingDim,
                activation: 'linear',
                kernelInitializer: 'glorotNormal',
                name: 'item_output'
            }).apply(itemHidden);
            
            // Create tower models
            const userTower = tf.model({ 
                inputs: userInput, 
                outputs: userOutput 
            });
            
            const itemTower = tf.model({ 
                inputs: [itemInput, genreInput], 
                outputs: itemOutput 
            });
            
            // Combined model
            const combinedUserInput = tf.input({ shape: [1], dtype: 'int32' });
            const combinedItemInput = tf.input({ shape: [1], dtype: 'int32' });
            const combinedGenreInput = tf.input({ shape: [this.numGenres], dtype: 'float32' });
            
            const userTowerOutput = userTower.apply(combinedUserInput);
            const itemTowerOutput = itemTower.apply([combinedItemInput, combinedGenreInput]);
            const dotProduct = tf.layers.dot({ axes: 1, normalize: false }).apply([userTowerOutput, itemTowerOutput]);
            
            const combinedModel = tf.model({
                inputs: [combinedUserInput, combinedItemInput, combinedGenreInput],
                outputs: dotProduct
            });
            
            console.log('MLPTwoTower model created successfully');
            return combinedModel;
            
        } catch (error) {
            console.error('Error creating MLPTwoTower model:', error);
            throw error;
        }
    }
}

class DeepLearningTwoTower extends TwoTowerBase {
    createModel() {
        try {
            // User tower: deep architecture
            const userInput = tf.input({ shape: [1], dtype: 'int32', name: 'user_input' });
            const userEmbedding = tf.layers.embedding({
                inputDim: this.numUsers,
                outputDim: this.embeddingDim * 2,
                embeddingsInitializer: 'glorotNormal',
                name: 'user_embedding'
            }).apply(userInput);
            const userFlatten = tf.layers.flatten().apply(userEmbedding);
            
            // Deep MLP for user tower
            const userHidden1 = tf.layers.dense({
                units: 128,
                activation: 'relu',
                kernelInitializer: 'heNormal',
                name: 'user_hidden1'
            }).apply(userFlatten);
            
            const userHidden2 = tf.layers.dense({
                units: 64,
                activation: 'relu',
                kernelInitializer: 'heNormal',
                name: 'user_hidden2'
            }).apply(userHidden1);
            
            const userOutput = tf.layers.dense({
                units: this.embeddingDim,
                activation: 'linear',
                kernelInitializer: 'glorotNormal',
                name: 'user_output'
            }).apply(userHidden2);
            
            // Item tower: deep architecture with genre features
            const itemInput = tf.input({ shape: [1], dtype: 'int32', name: 'item_input' });
            const genreInput = tf.input({ shape: [this.numGenres], dtype: 'float32', name: 'genre_input' });
            
            const itemEmbedding = tf.layers.embedding({
                inputDim: this.numItems,
                outputDim: this.embeddingDim * 2,
                embeddingsInitializer: 'glorotNormal',
                name: 'item_embedding'
            }).apply(itemInput);
            const itemFlatten = tf.layers.flatten().apply(itemEmbedding);
            
            // Combine and deep process
            const itemConcat = tf.layers.concatenate().apply([itemFlatten, genreInput]);
            const itemHidden1 = tf.layers.dense({
                units: 128,
                activation: 'relu',
                kernelInitializer: 'heNormal',
                name: 'item_hidden1'
            }).apply(itemConcat);
            
            const itemHidden2 = tf.layers.dense({
                units: 64,
                activation: 'relu',
                kernelInitializer: 'heNormal',
                name: 'item_hidden2'
            }).apply(itemHidden1);
            
            const itemOutput = tf.layers.dense({
                units: this.embeddingDim,
                activation: 'linear',
                kernelInitializer: 'glorotNormal',
                name: 'item_output'
            }).apply(itemHidden2);
            
            // Create tower models
            const userTower = tf.model({ 
                inputs: userInput, 
                outputs: userOutput 
            });
            
            const itemTower = tf.model({ 
                inputs: [itemInput, genreInput], 
                outputs: itemOutput 
            });
            
            // Combined model
            const combinedUserInput = tf.input({ shape: [1], dtype: 'int32' });
            const combinedItemInput = tf.input({ shape: [1], dtype: 'int32' });
            const combinedGenreInput = tf.input({ shape: [this.numGenres], dtype: 'float32' });
            
            const userTowerOutput = userTower.apply(combinedUserInput);
            const itemTowerOutput = itemTower.apply([combinedItemInput, combinedGenreInput]);
            const dotProduct = tf.layers.dot({ axes: 1, normalize: false }).apply([userTowerOutput, itemTowerOutput]);
            
            const combinedModel = tf.model({
                inputs: [combinedUserInput, combinedItemInput, combinedGenreInput],
                outputs: dotProduct
            });
            
            console.log('DeepLearningTwoTower model created successfully');
            return combinedModel;
            
        } catch (error) {
            console.error('Error creating DeepLearningTwoTower model:', error);
            throw error;
        }
    }
}
