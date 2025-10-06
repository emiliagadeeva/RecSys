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

    async train(userInput, movieInput, ratings, epochs = 10, batchSize = 32) {
        if (!this.model) this.buildModel();

        console.log(`Training ${this.constructor.name}...`);
        const history = { loss: [], acc: [] };

        // Создаем тензоры для данных
        const movieIds = tf.tensor1d(movieInput.map(m => m.movieId), 'int32');
        const genres = tf.tensor2d(movieInput.map(m => m.genres));

        const numSamples = userInput.shape[0];
        console.log(`Training on ${numSamples} samples`);

        for (let epoch = 0; epoch < epochs; epoch++) {
            let epochLoss = 0;
            let epochAcc = 0;
            let batchCount = 0;

            // Создаем перемешанные индексы для каждой эпохи
            const indices = tf.util.createShuffledIndices(numSamples);
            const shuffledUserInput = userInput.gather(indices);
            const shuffledMovieIds = movieIds.gather(indices);
            const shuffledGenres = genres.gather(indices);
            const shuffledRatings = ratings.gather(indices);

            for (let i = 0; i < numSamples; i += batchSize) {
                const end = Math.min(i + batchSize, numSamples);
                
                const userBatch = shuffledUserInput.slice([i], [end - i]);
                const movieIdBatch = shuffledMovieIds.slice([i], [end - i]);
                const genreBatch = shuffledGenres.slice([i, 0], [end - i, -1]);
                const labelBatch = shuffledRatings.slice([i], [end - i]);

                try {
                    const result = await this.model.trainOnBatch(
                        [userBatch, movieIdBatch, genreBatch], 
                        labelBatch
                    );

                    epochLoss += result[0];
                    epochAcc += result[1];
                    batchCount++;

                } catch (error) {
                    console.error('Training batch error:', error);
                } finally {
                    // Очищаем тензоры батча
                    tf.dispose([userBatch, movieIdBatch, genreBatch, labelBatch]);
                }
            }

            // Очищаем перемешанные тензоры
            tf.dispose([shuffledUserInput, shuffledMovieIds, shuffledGenres, shuffledRatings]);

            if (batchCount > 0) {
                const avgLoss = epochLoss / batchCount;
                const avgAcc = epochAcc / batchCount;
                history.loss.push(avgLoss);
                history.acc.push(avgAcc);

                console.log(`Epoch ${epoch + 1}/${epochs} - loss: ${avgLoss.toFixed(4)} - acc: ${avgAcc.toFixed(4)}`);
                
                // Обновляем статус каждую эпоху
                if (typeof this.updateTrainingStatus === 'function') {
                    this.updateTrainingStatus(`Epoch ${epoch + 1}/${epochs} - Loss: ${avgLoss.toFixed(4)}`);
                }
            }
        }

        // Очищаем основные тензоры
        movieIds.dispose();
        genres.dispose();

        console.log(`Training completed. Final loss: ${history.loss[history.loss.length - 1]}`);
        return history;
    }

    async getUserEmbedding(userId) {
        if (!this.userTower) throw new Error('User tower not trained');
        
        const userInput = tf.tensor1d([userId], 'int32');
        const embedding = this.userTower.predict(userInput);
        userInput.dispose();
        return embedding;
    }

    async getItemEmbedding(movieId, genres) {
        if (!this.itemTower) throw new Error('Item tower not trained');
        
        const movieIdTensor = tf.tensor1d([movieId], 'int32');
        const genreTensor = tf.tensor2d([genres]);
        const embedding = this.itemTower.predict([movieIdTensor, genreTensor]);
        movieIdTensor.dispose();
        genreTensor.dispose();
        return embedding;
    }

    compileModel() {
        // Увеличиваем learning rate и используем правильные метрики
        this.model.compile({
            optimizer: tf.train.adam(0.01), // Увеличенный learning rate
            loss: 'binaryCrossentropy',
            metrics: ['accuracy', 'binaryCrossentropy']
        });
    }

    // Метод для обновления статуса обучения
    setStatusCallback(callback) {
        this.updateTrainingStatus = callback;
    }
}

class WithoutDLTwoTower extends BaseTwoTower {
    buildModel() {
        console.log('Building Without DL Two-Tower model...');
        
        // User Tower
        const userInput = tf.input({ shape: [1], dtype: 'int32' });
        const userEmbedding = tf.layers.embedding({
            inputDim: this.numUsers,
            outputDim: this.embeddingDim,
            embeddingsInitializer: 'randomNormal',
            name: 'user_embedding'
        }).apply(userInput);
        const userOutput = tf.layers.flatten().apply(userEmbedding);
        this.userTower = tf.model({ inputs: userInput, outputs: userOutput });

        // Item Tower
        const movieIdInput = tf.input({ shape: [1], dtype: 'int32' });
        const genreInput = tf.input({ shape: [this.numGenres] });
        
        const movieEmbedding = tf.layers.embedding({
            inputDim: this.numItems,
            outputDim: this.embeddingDim,
            embeddingsInitializer: 'randomNormal',
            name: 'movie_embedding'
        }).apply(movieIdInput);
        const movieFlatten = tf.layers.flatten().apply(movieEmbedding);
        
        // Genre processing
        const genreProjection = tf.layers.dense({
            units: this.embeddingDim,
            activation: 'linear',
            name: 'genre_projection'
        }).apply(genreInput);
        
        // Combine movie and genre features
        const itemOutput = tf.layers.add().apply([movieFlatten, genreProjection]);
        this.itemTower = tf.model({ inputs: [movieIdInput, genreInput], outputs: itemOutput });

        // Combined model
        const userVec = this.userTower.apply(userInput);
        const itemVec = this.itemTower.apply([movieIdInput, genreInput]);
        
        // Dot product with temperature scaling
        const dotProduct = tf.layers.dot({ axes: 1 }).apply([userVec, itemVec]);
        
        // Scale dot product to better range
        const scaled = tf.layers.dense({
            units: 1,
            activation: 'linear',
            kernelInitializer: 'ones',
            useBias: false
        }).apply(dotProduct);
        
        const prediction = tf.layers.dense({ 
            units: 1, 
            activation: 'sigmoid',
            name: 'prediction'
        }).apply(scaled);

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
        
        // User Tower
        const userInput = tf.input({ shape: [1], dtype: 'int32' });
        const userEmbedding = tf.layers.embedding({
            inputDim: this.numUsers,
            outputDim: 32,
            embeddingsInitializer: 'randomNormal'
        }).apply(userInput);
        const userFlatten = tf.layers.flatten().apply(userEmbedding);
        
        // MLP with dropout for regularization
        const userHidden = tf.layers.dense({
            units: 24,
            activation: 'relu'
        }).apply(userFlatten);
        
        const userOutput = tf.layers.dense({
            units: this.embeddingDim,
            activation: 'tanh' // tanh для ограничения значений
        }).apply(userHidden);
        
        this.userTower = tf.model({ inputs: userInput, outputs: userOutput });

        // Item Tower
        const movieIdInput = tf.input({ shape: [1], dtype: 'int32' });
        const genreInput = tf.input({ shape: [this.numGenres] });
        
        const movieEmbedding = tf.layers.embedding({
            inputDim: this.numItems,
            outputDim: 32,
            embeddingsInitializer: 'randomNormal'
        }).apply(movieIdInput);
        const movieFlatten = tf.layers.flatten().apply(movieEmbedding);
        
        const genreHidden = tf.layers.dense({
            units: 16,
            activation: 'relu'
        }).apply(genreInput);
        
        const combined = tf.layers.concatenate().apply([movieFlatten, genreHidden]);
        
        const itemHidden = tf.layers.dense({
            units: 24,
            activation: 'relu'
        }).apply(combined);
        
        const itemOutput = tf.layers.dense({
            units: this.embeddingDim,
            activation: 'tanh'
        }).apply(itemHidden);
        
        this.itemTower = tf.model({ inputs: [movieIdInput, genreInput], outputs: itemOutput });

        // Combined model
        const userVec = this.userTower.apply(userInput);
        const itemVec = this.itemTower.apply([movieIdInput, genreInput]);
        
        const dotProduct = tf.layers.dot({ axes: 1 }).apply([userVec, itemVec]);
        
        const prediction = tf.layers.dense({ 
            units: 1, 
            activation: 'sigmoid'
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
        
        // User Tower
        const userInput = tf.input({ shape: [1], dtype: 'int32' });
        const userEmbedding = tf.layers.embedding({
            inputDim: this.numUsers,
            outputDim: 64,
            embeddingsInitializer: 'randomNormal'
        }).apply(userInput);
        const userFlatten = tf.layers.flatten().apply(userEmbedding);
        
        const userDense1 = tf.layers.dense({
            units: 48,
            activation: 'relu'
        }).apply(userFlatten);
        
        const userDense2 = tf.layers.dense({
            units: 32,
            activation: 'relu'
        }).apply(userDense1);
        
        const userOutput = tf.layers.dense({
            units: this.embeddingDim,
            activation: 'tanh'
        }).apply(userDense2);
        
        this.userTower = tf.model({ inputs: userInput, outputs: userOutput });

        // Item Tower
        const movieIdInput = tf.input({ shape: [1], dtype: 'int32' });
        const genreInput = tf.input({ shape: [this.numGenres] });
        
        const movieEmbedding = tf.layers.embedding({
            inputDim: this.numItems,
            outputDim: 64,
            embeddingsInitializer: 'randomNormal'
        }).apply(movieIdInput);
        const movieFlatten = tf.layers.flatten().apply(movieEmbedding);
        
        const genreDense1 = tf.layers.dense({
            units: 32,
            activation: 'relu'
        }).apply(genreInput);
        
        const genreDense2 = tf.layers.dense({
            units: 24,
            activation: 'relu'
        }).apply(genreDense1);
        
        const combined = tf.layers.concatenate().apply([movieFlatten, genreDense2]);
        
        const itemDense1 = tf.layers.dense({
            units: 48,
            activation: 'relu'
        }).apply(combined);
        
        const itemDense2 = tf.layers.dense({
            units: 32,
            activation: 'relu'
        }).apply(itemDense1);
        
        const itemOutput = tf.layers.dense({
            units: this.embeddingDim,
            activation: 'tanh'
        }).apply(itemDense2);
        
        this.itemTower = tf.model({ inputs: [movieIdInput, genreInput], outputs: itemOutput });

        // Combined model
        const userVec = this.userTower.apply(userInput);
        const itemVec = this.itemTower.apply([movieIdInput, genreInput]);
        
        const dotProduct = tf.layers.dot({ axes: 1 }).apply([userVec, itemVec]);
        
        const prediction = tf.layers.dense({ 
            units: 1, 
            activation: 'sigmoid'
        }).apply(dotProduct);

        this.model = tf.model({
            inputs: [userInput, movieIdInput, genreInput],
            outputs: prediction
        });

        this.compileModel();
        console.log('Deep Learning Two-Tower model built successfully');
    }
}
