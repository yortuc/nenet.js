var v = require('vectorious'), Matrix = v.Matrix, Vector = v.Vector, BLAS = v.BLAS; // access BLAS routines
var Nenet = require('../../src/nenet'),
	utils = require('../../src/util');

// train
const data = utils.parse_data(utils.read_file('./boston_normalized.txt'), 13);
//, Nenet.funcs.linear_activation, Nenet.funcs.linear_activation_grad

const nn = new Nenet([
		["input",  13],
		["hidden", 100],
		["output", 1]
	])
	.options({
		dataSet: data,
		miniBatchSize: 120,
		learningRate: 0.01
	})
	.train(15000);

// training accuracy
const y_pred = nn.activation(data.x);
const training_error = Nenet.funcs.quadratic(new Matrix(data.y), new Matrix(y_pred));

console.log("training error: ", training_error);

utils.log_file(y_pred[0].reduce((prev,cur)=> cur + "\n" + prev ,""), "y_pred.txt");

//var correct_estimates = utils.classification_error(y_pred, data.y);
//console.log("training accuracy: %", 100 * correct_estimates / data.x[0].length);

// validation accuracy
/*
const validation_data = utils.parse_data(utils.read_file('./iris_data_validation.txt'), 4);
const validation_estimate = nn.activation(validation_data.x);
const validation_accuracy = utils.classification_error(validation_estimate, validation_data.y) * 100.0 / validation_data.x[0].length;

console.log("validation accuracy: %", validation_accuracy);
*/