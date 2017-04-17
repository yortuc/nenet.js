var Nenet = require('../../src/nenet'),
	utils = require('../../src/util');

// train
const iris_training_data = utils.parse_data(utils.read_file('./iris_data_training.txt'), 4);

const nn = new Nenet([
		["input",  4],
		["hidden", 6],
		["output", 3]
	])
	.options({
		dataSet: iris_training_data,
		miniBatchSize: 25,
		learningRate: 0.2
	})
	.train(5000);

// training accuracy
const y_pred = nn.activation(iris_training_data.x);
var correct_estimates = utils.classification_error(y_pred, iris_training_data.y);
console.log("training accuracy: %", 100 * correct_estimates / iris_training_data.x[0].length);

// validation accuracy
const validation_data = utils.parse_data(utils.read_file('./iris_data_validation.txt'), 4);
const validation_estimate = nn.activation(validation_data.x);
const validation_accuracy = utils.classification_error(validation_estimate, validation_data.y) * 100.0 / validation_data.x[0].length;

console.log("validation accuracy: %", validation_accuracy);