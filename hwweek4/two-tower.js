class TwoTowerBase {
    constructor(numUsers, numItems, embeddingDim, numGenres) {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.embeddingDim = embeddingDim;
        this.numGenres = numGenres;
        this.model = null;
        this.optimizer = tf.train.adam(0.001);
    }

    createModel() {
        throw new Error('createModel must be implemented by subclass');
    }

    async train(ratings, movies, epochs = 5, batchSize = 64) {
        if (!this.model) {
            this.model = this.createModel();
        }

        // Prepare training data
        const { userInput, itemInput, genreFeatures, labels } = this.prepareTrainingData(ratings, movies);

        const history = { loss: [] };

        for (let epoch = 0; epoch < epochs; epoch++) {
            const epochLoss = await this.trainEpoch(userInput, itemInput, genreFeatures, labels, batchSize);
            history.loss.push(epochLoss);
            
            console.log(`Epoch ${epoch + 1}/${epochs}, Loss: ${epochLoss.toFixed(4)}`);
            
            // Update UI periodically
            if (epoch % 2 === 0) {
                await tf.nextFrame();
            }
        }

        // Clean up tensors
        tf.dispose([userInput, itemInput, genreFeatures, labels]);

        return history;
    }

    prepareTrainingData(ratings, movies) {
        const userInput = [];
        const itemInput = [];
        const labels = [];

        for (const rating of ratings) {
            userInput.push(rating.user_id - 1); // Convert to 0-based index
            itemInput.push(rating.item_id - 1);
            labels.push(rating.rating >= 4 ? 1 : 0); // Binary labels for retrieval
        }

        // Create genre features tensor
        const genreFeaturesArray = [];
        for (const itemId of itemInput) {
            const movie = movies[itemId + 1];
            if (movie && movie.genreFeatures) {
                genreFeaturesArray.push(movie.genreFeatures);
            } else {
                // Default genre features if movie not found
                genreFeaturesArray.push(Array(this.numGenres).fill(0));
            }
        }

        return {
            userInput: tf.tensor1d(userInput, 'int32'),
            itemInput: tf.tensor1d(itemInput, 'int32'),
            genreFeatures: tf.tensor2d(genreFeaturesArray, [genreFeaturesArray.length, this.numGenres], 'float32'),
            labels: tf.tensor1d(labels, 'float32')
        };
    }

    async trainEpoch(userInput, itemInput, genreFeatures, labels, batchSize) {
        const userArray = await userInput.array();
        const itemArray = await itemInput.array();
        const genreArray = await genreFeatures.array();
        const labelArray = await labels.array();

        const numBatches = Math.ceil(userArray.length / batchSize);
        let totalLoss = 0;

        for (let i = 0; i < numBatches; i++) {
            const start = i * batchSize;
            const end = Math.min(start + batchSize, userArray.length);
            
            const batchUser = tf.tensor1d(userArray.slice(start, end), 'int32');
            const batchItem = tf.tensor1d(itemArray.slice(start, end), 'int32');
            const batchGenre = tf.tensor2d(genreArray.slice(start, end), [end - start, this.numGenres], 'float32');
            const batchLabel = tf.tensor1d(labelArray.slice(start, end), 'float32');

            const loss = this.trainStep(batchUser, batchItem, batchGenre, batchLabel);
            totalLoss += loss;

            tf.dispose([batchUser, batchItem, batchGenre, batchLabel]);
            
            // Allow UI updates
            if (i % 10 === 0) {
                await tf.nextFrame();
            }
        }

        return totalLoss / numBatches;
    }

    trainStep(userIds, itemIds, genreFeatures, labels) {
        return tf.tidy(() => {
            const lossFn = () => {
                const userEmbeddings = this.model.layers[0].apply(userIds);
                const itemEmbeddings = this.model.layers[1].apply([itemIds, genreFeatures]);
                
                const scores = tf.sum(tf.mul(userEmbeddings, itemEmbeddings), 1);
                const predictions = tf.sigmoid(scores);
                
                return tf.losses.sigmoidCrossEntropy(labels, predictions);
            };
            
            const lossValue = this.optimizer.minimize(lossFn, true);
            return lossValue ? lossValue.dataSync()[0] : 0;
        });
    }

    async recommend(userId, movies, topK = 10) {
        if (!this.model) {
            throw new Error('Model not trained');
        }

        return tf.tidy(() => {
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

            try {
                // Get user embedding
                const userEmbedding = this.model.layers[0].apply(userTensor);
                
                // Get all item embeddings
                const itemEmbeddings = this.model.layers[1].apply([itemTensor, genreTensor]);
                
                // Calculate scores
                const userEmbeddingRepeated = userEmbedding.tile([this.numItems, 1]);
                const scores = tf.sum(tf.mul(userEmbeddingRepeated, itemEmbeddings), 1);
                
                // Get top K recommendations
                const { values, indices } = tf.topk(scores, topK);
                
                const topScores = values.arraySync();
                const topIndices = indices.arraySync();
                
                const recommendations = [];
                for (let i = 0; i < topK; i++) {
                    const itemId = topIndices[i] + 1;
                    if (movies[itemId]) {
                        recommendations.push({
                            id: itemId,
                            title: movies[itemId].title,
                            score: topScores[i],
                            genres: movies[itemId].genres || ['Unknown']
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
            } finally {
                tf.dispose([userTensor, itemTensor, genreTensor]);
            }
        });
    }
}

class WithoutDLTwoTower extends TwoTowerBase {
    createModel() {
        // User tower: simple embedding
        const userInput = tf.input({ shape: [1], dtype: 'int32', name: 'user_input' });
        const userEmbedding = tf.layers.embedding({
            inputDim: this.numUsers,
            outputDim: this.embeddingDim,
            embeddingsInitializer: 'glorotNormal',
            name: 'user_embedding'
        }).apply(userInput);
        const userFlatten = tf.layers.flatten().apply(userEmbedding);
        
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
        const itemGenreConcat = tf.layers.concatenate().apply([itemFlatten, genreInput]);
        const itemProjection = tf.layers.dense({
            units: this.embeddingDim,
            activation: 'linear',
            kernelInitializer: 'glorotNormal',
            name: 'item_projection'
        }).apply(itemGenreConcat);
        
        const userTower = tf.model({ inputs: userInput, outputs: userFlatten });
        const itemTower = tf.model({ 
            inputs: [itemInput, genreInput], 
            outputs: itemProjection 
        });
        
        // Combined model for training
        const combinedInput = [userInput, itemInput, genreInput];
        const userOutput = userTower.apply(userInput);
        const itemOutput = itemTower.apply([itemInput, genreInput]);
        const dotProduct = tf.layers.dot({ axes: 1, normalize: false }).apply([userOutput, itemOutput]);
        
        const model = tf.model({
            inputs: combinedInput,
            outputs: dotProduct
        });
        
        return model;
    }
}

class MLPTwoTower extends TwoTowerBase {
    constructor(numUsers, numItems, embeddingDim, numGenres, userFeatures = null) {
        super(numUsers, numItems, embeddingDim, numGenres);
        this.userFeatures = userFeatures;
    }

    createModel() {
        // User tower: embedding + MLP
        const userInput = tf.input({ shape: [1], dtype: 'int32', name: 'user_input' });
        const userEmbedding = tf.layers.embedding({
            inputDim: this.numUsers,
            outputDim: this.embeddingDim,
            embeddingsInitializer: 'glorotNormal',
            name: 'user_embedding'
        }).apply(userInput);
        const userFlatten = tf.layers.flatten().apply(userEmbedding);
        
        // MLP layers for user tower
        const userHidden1 = tf.layers.dense({
            units: 64,
            activation: 'relu',
            kernelInitializer: 'heNormal',
            name: 'user_hidden1'
        }).apply(userFlatten);
        
        const userOutput = tf.layers.dense({
            units: this.embeddingDim,
            activation: 'linear',
            kernelInitializer: 'glorotNormal',
            name: 'user_output'
        }).apply(userHidden1);
        
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
        
        // Combine item embedding with genre features
        const itemGenreConcat = tf.layers.concatenate().apply([itemFlatten, genreInput]);
        
        // MLP layers for item tower
        const itemHidden1 = tf.layers.dense({
            units: 64,
            activation: 'relu',
            kernelInitializer: 'heNormal',
            name: 'item_hidden1'
        }).apply(itemGenreConcat);
        
        const itemOutput = tf.layers.dense({
            units: this.embeddingDim,
            activation: 'linear',
            kernelInitializer: 'glorotNormal',
            name: 'item_output'
        }).apply(itemHidden1);
        
        const userTower = tf.model({ inputs: userInput, outputs: userOutput });
        const itemTower = tf.model({ 
            inputs: [itemInput, genreInput], 
            outputs: itemOutput 
        });
        
        // Combined model for training
        const combinedInput = [userInput, itemInput, genreInput];
        const userTowerOutput = userTower.apply(userInput);
        const itemTowerOutput = itemTower.apply([itemInput, genreInput]);
        const dotProduct = tf.layers.dot({ axes: 1, normalize: false }).apply([userTowerOutput, itemTowerOutput]);
        
        const model = tf.model({
            inputs: combinedInput,
            outputs: dotProduct
        });
        
        return model;
    }
}

class DeepLearningTwoTower extends TwoTowerBase {
    constructor(numUsers, numItems, embeddingDim, numGenres, userFeatures = null) {
        super(numUsers, numItems, embeddingDim, numGenres);
        this.userFeatures = userFeatures;
    }

    createModel() {
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
        
        // Combine item embedding with genre features
        const itemGenreConcat = tf.layers.concatenate().apply([itemFlatten, genreInput]);
        
        // Deep MLP for item tower
        const itemHidden1 = tf.layers.dense({
            units: 128,
            activation: 'relu',
            kernelInitializer: 'heNormal',
            name: 'item_hidden1'
        }).apply(itemGenreConcat);
        
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
        
        const userTower = tf.model({ inputs: userInput, outputs: userOutput });
        const itemTower = tf.model({ 
            inputs: [itemInput, genreInput], 
            outputs: itemOutput 
        });
        
        // Combined model for training
        const combinedInput = [userInput, itemInput, genreInput];
        const userTowerOutput = userTower.apply(userInput);
        const itemTowerOutput = itemTower.apply([itemInput, genreInput]);
        const dotProduct = tf.layers.dot({ axes: 1, normalize: false }).apply([userTowerOutput, itemTowerOutput]);
        
        const model = tf.model({
            inputs: combinedInput,
            outputs: dotProduct
        });
        
        return model;
    }
}
