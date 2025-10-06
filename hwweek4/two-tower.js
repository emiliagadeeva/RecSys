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

    async train(userInput, movieInput, ratings, epochs = 3, batchSize = 64) {
        if (!this.model) this.buildModel();

        console.log(`Training ${this.constructor.name}...`);
        const history = { loss: [], acc: [] };

        const movieIds = tf.tensor1d(movieInput.map(m => m.movieId), 'int32');
        const genres = tf.tensor2d(movieInput.map(m => m.genres));

        for (let epoch = 0; epoch < epochs; epoch++) {
            let epochLoss = 0;
            let batchCount = 0;

            for (let i = 0; i < userInput.shape[0]; i += batchSize) {
                const end = Math.min(i + batchSize, userInput.shape[0]);
                
                const userBatch = userInput.slice([i], [end - i]);
                const movieIdBatch = movieIds.slice([i], [end - i]);
                const genreBatch = genres.slice([i, 0], [end - i, -1]);
                const labelBatch = ratings.slice([i], [end - i]);

                const result = await this.model.trainOnBatch(
                    [userBatch, movieIdBatch, genreBatch], 
                    labelBatch
                );

                epochLoss += result[0];
                batchCount++;

                tf.dispose([userBatch, movieIdBatch, genreBatch, labelBatch]);
            }

            const avgLoss = epochLoss / batchCount;
            history.loss.push(avgLoss);
            console.log(`Epoch ${epoch + 1}/${epochs} - loss: ${avgLoss.toFixed(4)}`);
        }

        movieIds.dispose();
        genres.dispose();

        return history;
    }

    async getUserEmbedding(userId) {
        const userInput = tf.tensor1d([userId], 'int32');
        const embedding = this.userTower.predict(userInput);
        userInput.dispose();
        return embedding;
    }

    async getItemEmbedding(movieId, genres) {
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

class WithoutDLTwoTower extends BaseTwoTower {
    buildModel() {
        console.log('Building Without DL Two-Tower model...');
        
        // User Tower: Simple embedding
        const userInput = tf.input({ shape: [1], dtype: 'int32' });
        const userEmbedding = tf.layers.embedding({
            inputDim: this.numUsers,
            outputDim: this.embeddingDim
        }).apply(userInput);
        const userOutput = tf.layers.flatten().apply(userEmbedding);
        this.userTower = tf.model({ inputs: userInput, outputs: userOutput });

        // Item Tower: Embedding + Genre features
        const movieIdInput = tf.input({ shape: [1], dtype: 'int32' });
        const genreInput = tf.input({ shape: [this.numGenres] });
        
        const movieEmbedding = tf.layers.embedding({
            inputDim: this.numItems,
            outputDim: this.embeddingDim
        }).apply(movieIdInput);
        const movieFlatten = tf.layers.flatten().apply(movieEmbedding);
        
        // Project genres to embedding dimension and add to movie embedding
        const genreProjection = tf.layers.dense({
            units: this.embeddingDim,
            activation: 'linear'
        }).apply(genreInput);
        
        const itemOutput = tf.layers.add().apply([movieFlatten, genreProjection]);
        this.itemTower = tf.model({ inputs: [movieIdInput, genreInput], outputs: itemOutput });

        // Combined model
        const userVec = this.userTower.apply(userInput);
        const itemVec = this.itemTower.apply([movieIdInput, genreInput]);
        const dotProduct = tf.layers.dot({ axes: 1 }).apply([userVec, itemVec]);
        const prediction = tf.layers.dense({ units: 1, activation: 'sigmoid' }).apply(dotProduct);

        this.model = tf.model({
            inputs: [userInput, movieIdInput, genreInput],
            outputs: prediction
        });

        this.compileModel();
    }
}

class MLPTwoTower extends BaseTwoTower {
    buildModel() {
        console.log('Building MLP Two-Tower model...');
        
        // User Tower: Embedding + MLP
        const userInput = tf.input({ shape: [1], dtype: 'int32' });
        const userEmbedding = tf.layers.embedding({
            inputDim: this.numUsers,
            outputDim: 32
        }).apply(userInput);
        const userFlatten = tf.layers.flatten().apply(userEmbedding);
        
        // MLP with hidden layer (ReLU activation)
        const userHidden = tf.layers.dense({
            units: 24,
            activation: 'relu'
        }).apply(userFlatten);
        
        const userOutput = tf.layers.dense({
            units: this.embeddingDim,
            activation: 'linear'
        }).apply(userHidden);
        this.userTower = tf.model({ inputs: userInput, outputs: userOutput });

        // Item Tower: Embedding + Genres + MLP
        const movieIdInput = tf.input({ shape: [1], dtype: 'int32' });
        const genreInput = tf.input({ shape: [this.numGenres] });
        
        const movieEmbedding = tf.layers.embedding({
            inputDim: this.numItems,
            outputDim: 32
        }).apply(movieIdInput);
        const movieFlatten = tf.layers.flatten().apply(movieEmbedding);
        
