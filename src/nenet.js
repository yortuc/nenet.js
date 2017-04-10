var v = require('vectorious'), Matrix = v.Matrix, Vector = v.Vector, BLAS = v.BLAS; // access BLAS routines

const sigmoid = (z)=> 1.0 / (1.0 + Math.exp(-z));		// sgm(z) = 1 / (1 + exp(-z))
const sigmoid_grad = (a)=> a.product( a.map(c=> 1-c) );	// d(sgm)/da = a * (1-a)
const quadratic = (y, a) => {
	const sb = Matrix.subtract(y, a);
	const sb2 = sb.product(sb);
	const accum = sb2.reduce((prev, cur)=> prev+cur);
	return 1/(2 * y.shape[1]) * accum;
}		
const cross_entrophy = function(target, output) {
	var crossentropy = 0;
	for (var i in output)
	  crossentropy -= (target[i] * Math.log(output[i]+1e-15)) + ((1-target[i]) * Math.log((1+1e-15)-output[i])); // +1e-15 is a tiny push away to avoid Math.log(0)
	return crossentropy;
}
function shuffle_arrays(x, y){
	var currentIndex = x.length, tmp1, tmp2, randomIndex;

	// While there remain elements to shuffle...
	while (0 !== currentIndex) {
		// Pick a remaining element...
		randomIndex = Math.floor(Math.random() * currentIndex);
		currentIndex -= 1;

		// And swap it with the current element.
		tmp1 = x[currentIndex];
		tmp2 = y[currentIndex];
		x[currentIndex] = x[randomIndex];
		y[currentIndex] = y[randomIndex];
		x[randomIndex] = tmp1;
		y[randomIndex] = tmp2;
	}	
	return {x, y};
}

// add given column-matrix to every element of a row in given matrix
function addToRows(matrix, b) {
	const rows = matrix.shape[0];
	const cols = matrix.shape[1];
	matrix = matrix.toArray();
	const matrixSubset = [];

	for(var i=0; i<rows; i++){
		matrixSubset.push([]);
		for(var j=0; j<cols; j++){
			matrixSubset[i].push(matrix[i][j] + b.get(i,0));
		}
	}

	return new Matrix(matrixSubset);
}

// inp [1;2;3;4] colSize:3 -> [1,1,1;2,2,2;3,3,3]
function fillRows(colMatrix, colSize) {
	const rows = colMatrix.shape[0];
	const arr = colMatrix.toArray();
	const ret = [];

	for(var i=0; i<rows; i++){
		ret.push([]);
		const rowVal = arr[i][0];
		for(var j=0; j<colSize; j++){
			ret[i].push(rowVal);
		}
	}

	return new Matrix(ret);
}

class Layer {
	constructor(size) {
		this.size = size;
		this.type = null;
		this.z = new Matrix.zeros(size[0], 1);
		this.a = new Matrix(size[0], 1);
	}

	__initWeights(data_size) {
		const bCol = new Matrix.random(this.size[0], 1, 0, -0.5);
		this.b = fillRows(bCol, data_size);

		this.w = new Matrix.random(...this.size, 0, -0.5);
		this.del = null;
		this.w_grad = null;
		return this;
	}

	log() { console.log({z: this.z.toArray(), a: this.a.toArray() }); }
}

class Nenet {
	constructor(opt) {
		this.x = null;			// input matrix
		this.y = null;			// output matrix
		this.y_pred = null;		// estimation
		this.layers = [];		// layers array
		this.err = null;		// system error
		this.e = 0.05;			// learning rate

		this.errorFunction = Nenet.funcs.quadric;
		this.onIterationStep = 10;
		this.onIteration = ((step)=> console.log("i: " + step, this.err)).bind(this);

		this._createLayers(opt);
	}

	options(opt) {
		this.errorFunction = opt.errorFunction || this.errorFunction;
		this.e = opt.learningRate || this.e;
		this.onIterationStep = opt.onIterationStep || this.onIterationStep;
		this.onIteration = opt.onIteration || this.onIteration;

		return this;
	}

	_createLayers(opt) {
		// create layers
		for(var i=0; i<opt.length; i++){
			let cur_layer = opt[i];
			let prev_layer = i > 0 ? this.layers[i-1] : null;
			let size;
			let lyr;
			let xSize;

			switch(cur_layer[0]	/*layer type*/) {
				case "input":
					this.x = new Matrix(cur_layer[1]);
					size = [cur_layer[1].length];
					this.xSize = this.x.shape[1];
					lyr = new Layer(size);
					lyr.a = this.x;
					break;

				case "hidden":
					size = [cur_layer[1], prev_layer.size[0]];
					lyr = new Layer(size).__initWeights(this.xSize);
					break;

				case "output":
					this.y = new Matrix(cur_layer[1]);
					size = [cur_layer[1].length, prev_layer.size[0]];
					lyr = new Layer(size).__initWeights(this.xSize);
					break;
			}
			lyr.type = cur_layer[0];
			this.layers.push(lyr);
		}
	}

	log() { this.layers.forEach(l=> l.log()); return this; }

	train(max_iter) {
		for(var i=0; i<max_iter; i++){
			this.iterate_training(i, max_iter);
		}

		return this;
	}

	iterate_training(step, max) {
		this.feedForwardNetwork();

		// system error
		this.err = quadratic(this.y, this.layers[this.layers.length-1].a);

		if(step % this.onIterationStep === 0){
			this.onIteration(step);
		}

		if(step === max-1){
			this.y_pred = this.layers[this.layers.length-1].a.toArray();
		}

		/* (b) backpropagation */
		for(var j=this.layers.length-1; j>0; j--){
			this.backPropLayer(j);
		}
	}

	feedForwardNetwork() {
		// start from 1, exluding input layer
		for(var i=1; i<this.layers.length; i++){
			this.feedForwardLayer(i);
		}
	}

	feedForwardLayer(layer_index) {
		const prev_layer = this.layers[layer_index-1];
		const layer = this.layers[layer_index];

		layer.z = Matrix.multiply(layer.w, prev_layer.a);		
		layer.z = Matrix.add(layer.z, layer.b);
		layer.a = layer.z.map(sigmoid);
	}

	backPropLayer(layer_index) {
		const layer = this.layers[layer_index];
		const prev_layer = this.layers[layer_index-1];

		if(layer_index < this.layers.length-1){
			// non-output layer
			const next_layer = this.layers[layer_index+1];
			const bprop = Matrix.multiply(next_layer.w.T, next_layer.del);
			const net_grad = sigmoid_grad(layer.a);
			layer.del = Matrix.product(bprop, net_grad);
			layer.w_grad = Matrix.multiply(layer.del, prev_layer.a.T);
		}
		else{
			// output layer
			const output_err = Matrix.subtract(layer.a, this.y);
			const net_grad = sigmoid_grad(layer.a);

			layer.del = Matrix.product(output_err, net_grad);
			layer.w_grad = Matrix.multiply(layer.del, prev_layer.a.T);
		}
		
		layer.w = Matrix.subtract(layer.w, layer.w_grad.scale(this.e));
		const b_prod = Matrix.product(layer.b, layer.del.scale(this.e));
		layer.b = Matrix.subtract(layer.b, b_prod);
	}

	activation(xInput) {
		this.layers[0].a = new Matrix(xInput);
		this.feedForwardNetwork();
		return this.layers[this.layers.length-1].a.toArray();
	}

	logOutput() { 
		console.log({"i": "training ended", "err": this.err, "y_pred": this.y_pred});
		return this;
	}
}

Nenet.funcs = {
	sigmoid,
	sigmoid_grad,
	quadratic,
	cross_entrophy
};

module.exports = Nenet;