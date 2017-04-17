const Nenet = require('../../src/nenet'),
	  utils = require('../../src/util'); 

const mnist_data_raw = utils.read_file("./mnist_test.txt");

function parse_mnist_data(data){
	var x = [];
	var y = [];
	var data_col_size = 0;

	data.split('\n').map((row,i)=> {
		x.push([]);
		row.split('\t').map((col,j)=> {
			const val = parseInt(col);
			if(j===0){
				y.push(val);	// label
			}else{
				x[i].push(val/256.0);	// normalize pixel value
			}
		});

	});
	x = utils.mat_transpose(x);

	// convert labels to one-hot
	var y_hot = [];
	for(var i=0; i<y.length; i++){
		let label = y[i];
		let one_hot_arr = [0,0,0,0,0,0,0,0,0,0];
		one_hot_arr[label] = 1;
		y_hot.push(one_hot_arr);
	}
	y_hot = utils.mat_transpose(y_hot);

	return {x, y: y_hot};
}

const data = parse_mnist_data(mnist_data_raw);

const nn = new Nenet([
		["input", 784],	// 784 features (pixel values)
		["hidden", 100],	// 15 hidden layer neurons
		["output", 10]	// 10 classes
	])
	.options({ 
		dataSet: data,
		miniBatchSize: 20,
		errorFunction: Nenet.funcs.cross_entrophy,
		learningRate: 0.1,
		onIterationStep: 5
	})
	.train(5000);

// training accuracy
const y_pred = nn.activation(data.x);
var correct_estimates = utils.classification_error(y_pred, data.y);
console.log("training accuracy: %", 100 * correct_estimates / data.x[0].length);

// validation
const validation_data = parse_mnist_data(utils.read_file('./mnist_validation.txt'));
const validation_estimate = nn.activation(validation_data.x);
const validation_accuracy = utils.classification_error(validation_estimate, validation_data.y) * 100.0 / validation_data.x[0].length;
console.log("validation accuracy: %", validation_accuracy);

// save trained weights
const weights = utils.serialize_nn_weights(nn);
utils.log_file(JSON.stringify(weights), "./weights.txt");



