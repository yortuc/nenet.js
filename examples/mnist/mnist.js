const Nenet = require('../../src/nenet'),
	  utils = require('../../src/util'); 

const mnist_data_raw = utils.read_file("./mnist_min.txt");

function parse_data(data){
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

const data = parse_data(mnist_data_raw);

const nn = new Nenet([
		["input", 784],	// 784 features (pixel values)
		["hidden", 15],		// 15 
		["output", 10]	// 10 classes
	])
	.options({ 
		dataSet: data,
		miniBatchSize: 10,
		errorFunction: Nenet.funcs.cross_entrophy,
		learningRate: 0.2,
		onIterationStep: 5
	})
	.train(5000);

const y_pred = nn.activation(data.x);
utils.write_file(y_pred, "y_pred.txt");

var correct_estimates = utils.classification_error(y_pred, data.y);
console.log("training accuracy: %", 100 * correct_estimates / data.x[0].length);
