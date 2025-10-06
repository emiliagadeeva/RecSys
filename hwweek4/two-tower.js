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
            let batchResult = await iterator.next();
            
            while (!batchResult.done) {
                const { userBatch, movieIdBatch, genreBatch, labelBatch } = batchResult.value;
                
                const result = await this.model.trainOnBatch(
                    [userBatch, movieIdBatch, genreBatch], 
                    labelBatch
                );

                epochLoss += result[0];
                epochAcc += result[1];
                batchCount++;

                // Clean up tensors
                userBatch.dispose();
                movieIdBatch.dispose();
                genreBatch.dispose();
                labelBatch.dispose();

                batchResult = await iterator.next();
            }

            const avgLoss = epochLoss / batchCount;
            const avgAcc = epochAcc / batchCount;
            
            history.loss.push(avgLoss);
            history.acc.push(avgAcc);

            console.log(`Epoch ${epoch + 1}/${epochs} - loss: ${avgLoss.toFixed(4)} - acc: ${avgAcc.toFixed(4)}`);
        }

        // Clean up input tensors
        userInput.dispose();
        ratings.dispose();

        return history;
    }

    createDataset(userInput, movieInput, ratings, batchSize) {
        return {
            iterator: () => {
                let position = 0;
                const dataSize = userInput.shape[0];
                
                return {
                    next: async () => {
                        if (position >= dataSize) {
                            return { done: true };
                        }

                        const end = Math.min(position + batchSize, dataSize);
                        
                        const userBatch = userInput.slice([position], [end - position]);
                        const movieIdBatch = tf.tensor1d(
                            movieInput.slice(position, end - position).map(m => m.movieId),
                            'int32'
                        );
                        const genreBatch = tf.tensor2d(
                            movieInput.slice(position, end - position).map(m => m.genres)
                        );
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
        
        // User tower - simple embedding
        const userInput = tf.input({ shape: [1], name: 'user_input', dtype: 'int32' });
        const userEmbedding = tf.layers.embedding({
            inputDim: this.numUsers,
            outputDim: this.embeddingDim,
            name: 'user_embedding'
        }).apply(userInput);
        const userFlatten = tf.layers.flatten().apply(userEmbedding);
        this.userTower = tf.model({ inputs: userInput, outputs: userFl
