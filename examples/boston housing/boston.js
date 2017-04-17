var Nenet = require('../../src/nenet'),
	utils = require('../../src/util');

// train
const data = utils.parse_data(utils.read_file('./boston_normalized.txt'), 13);

console.log(data.x.length);
console.log(data.y.length);

const nn = new Nenet([
		["input",  13],
		["hidden", 6],
		["output", 1]
	])
	.options({
		dataSet: data,
		miniBatchSize: 20,
		learningRate: 0.01
	})
	.train(5000);

// training accuracy
const y_pred = nn.activation(data.x);

console.log(y_pred);

//var correct_estimates = utils.classification_error(y_pred, data.y);
//console.log("training accuracy: %", 100 * correct_estimates / data.x[0].length);

// validation accuracy
/*
const validation_data = utils.parse_data(utils.read_file('./iris_data_validation.txt'), 4);
const validation_estimate = nn.activation(validation_data.x);
const validation_accuracy = utils.classification_error(validation_estimate, validation_data.y) * 100.0 / validation_data.x[0].length;

console.log("validation accuracy: %", validation_accuracy);
*/