var Nenet = require('../../src/nenet'),
	utils = require('../../src/util');

// train
const {x, y} = utils.parse_data(utils.read_file('./iris_training.txt'), 4);

const nn = new Nenet([
		["input", x],
		["hidden", 6],
		["output", y]
	])
	.train(500);

var correct_estimates = utils.classification_error(nn.y_pred, y);
console.log("training accuracy: %", 100 * correct_estimates / x[0].length);

// validation
const validation_data = utils.parse_data(utils.read_file('./iris_training.txt'), 4);
const validation_estimate = nn.activation(validation_data.x);
const validation_accuracy = utils.classification_error(validation_estimate, validation_data.y) * 100.0 / validation_data.x[0].length;

console.log("validation accuracy: %", validation_accuracy);