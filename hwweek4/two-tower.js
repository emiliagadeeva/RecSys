class TwoTowerBase {
    constructor(numUsers, numItems, embeddingDim, numGenres) {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.embeddingDim = embeddingDim;
        this.numGenres = numGenres;
        this.model = null;
    }

    createModel() {
        throw new Error('createModel must be implemented by subclass');
    }

    async train(ratings, movies, epochs = 10, batchSize = 64) {
        if (!this.model) {
            this.model = this.createModel();
        }

        // Prepare training data
        const { userInput, itemInput, labels } = this.prepareTrainingData(ratings, movies);

        const history = { loss: [] };

        for (let epoch = 0; epoch < epochs; epoch++) {
            const epochLoss = await this.trainEpoch(userInput, itemInput, labels, batchSize);
            history.loss.push(epochLoss);
            
            console.log(`Epoch ${epoch + 1}/${epochs}, Loss: ${epochLoss.toFixed(4)}`);
            
            // Update UI periodically
            if (epoch % 2 === 0) {
                await tf.nextFrame();
            }
        }

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
        const genreFeatures = [];
        for (const itemId of itemInput) {
            const movie = movies[itemId + 1];
            genreFeatures.push(movie.genreFeatures);
        }

        return {
            userInput: tf.tensor1d(userInput, 'int32'),
            itemInput: tf.tensor1d(itemInput, 'int32'),
            genreFeatures: tf.tensor2d(genreFeatures, [genreFeatures.length, this.numGenres]),
            labels: tf.tensor1d(labels, 'float32')
        };
    }

    async trainEpoch(userInput, itemInput, labels, batchSize) {
        const dataset = tf.data.zip({
            user: tf.data.array(await userInput.array()),
            item: tf.data.array(await itemInput.array()),
            genres: tf.data.array(await this.genreFeatures.array()),
            label: tf.data.array(await labels.array())
        }).batch(batchSize).shuffle(1000);

        let totalLoss = 0;
        let batchCount = 0;

        await dataset.forEachAsync(batch => {
            const { user, item, genres, label } = batch;
            
            const loss = this.trainStep(user, item, genres, label);
            totalLoss += loss;
            batchCount++;
            
            tf.dispose([user, item, genres, label]);
        });

        return totalLoss / batchCount;
    }

    trainStep(userIds, itemIds, genreFeatures, labels) {
        return tf.tidy(() => {
            const optimizer = tf.train.adam(0.001);
            
            const loss = () => {
                const userEmbeddings = this.model.layers[0].apply(userIds);
                const itemEmbeddings = this.model.layers[1].apply([itemIds, genreFeatures]);
                
                const scores = tf.sum(tf.mul(userEmbeddings, itemEmbeddings), 1);
                const predictions = tf.sigmoid(scores);
                
                return tf.losses.sigmoidCrossEntropy(labels, predictions);
            };
            
            const lossValue = optimizer.minimize(loss, true).dataSync()[0];
            return lossValue;
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
                if (movies[i]) {
                    allGenreFeatures.push(movies[i].genreFeatures);
                } else {
                    allGenreFeatures.push(Array(this.numGenres).fill(0));
                }
            }
            const genreTensor = tf.tensor2d(allGenreFeatures, [this.numItems, this.numGenres]);

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
                        genres: movies[itemId].genres
                    });
                }
            }
            
            return recommendations;
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
        
        // Add user features if available
        let userFeatures = userFlatten;
        if (this.userFeatures) {
            // In a real implementation, you'd process user features here
            // For simplicity, we'll just use the embedding
        }
        
        // MLP layers for user tower
        const userHidden1 = tf.layers.dense({
            units: 64,
            activation: 'relu',
            kernelInitializer: 'heNormal',
            name: 'user_hidden1'
        }).apply(userFeatures);
        
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
            kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }),
            name: 'user_hidden1'
        }).apply(userFlatten);
        
        const userDropout1 = tf.layers.dropout({ rate: 0.3 }).apply(userHidden1);
        
        const userHidden2 = tf.layers.dense({
            units: 64,
            activation: 'relu',
            kernelInitializer: 'heNormal',
            kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }),
            name: 'user_hidden2'
        }).apply(userDropout1);
        
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
            kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }),
            name: 'item_hidden1'
        }).apply(itemGenreConcat);
        
        const itemDropout1 = tf.layers.dropout({ rate: 0.3 }).apply(itemHidden1);
        
        const itemHidden2 = tf.layers.dense({
            units: 64,
            activation: 'relu',
            kernelInitializer: 'heNormal',
            kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }),
            name: 'item_hidden2'
        }).apply(itemDropout1);
        
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
