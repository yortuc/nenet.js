/*
 * Utility functions 
 */

var fs = require('fs');

function read_file(filename) {
	var contents = fs.readFileSync(filename, 'utf8');
	return contents;
}

// takes first x_cols_size columns into x and takes rest into y
function parse_data(data, x_cols_size){
	var x = [];
	var y = [];
	var data_col_size = 0;

	data.split('\n').map((row,i)=> {
		if(i===0){
			let first_row = row + "";
			first_row.split('\t').forEach((col, index)=>{
				if(index<x_cols_size) x.push([]);
				if(index>=x_cols_size) y.push([]);
				data_col_size += 1;
			});
		}

		if(row.length>0){
 			row.split('\t').forEach((col, index)=> {
				if(index<x_cols_size) x[index].push(parseFloat(col));
				if(index>=x_cols_size) {
					y[index-x.length].push(parseFloat(col));
				}
			});
		}
	});

	return {x, y};
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

function test_shuffle_arrays(){
	var x = [1,2,3,4,5];
	var y = [2,4,6,8,10];
	console.log({x,y});
	var shuffled = shuffle_arrays(x,y);
	console.log(shuffled);
}

function mat_transpose(m){
	const ret = m[0].map((col, i)=> m.map(row=> row[i]));
	return ret;
}

function classification_error(y_pred, y) {
	var numCorrect = 0,
		numWrong = 0;

	y_pred = mat_transpose(y_pred);
	y = mat_transpose(y);

	y_pred.map((row,i)=> {
		let maxIndex = row.indexOf(Math.max(...row));

		if(y[i][maxIndex] === 0) {
			numWrong += 1;
		}
		else{
			numCorrect += 1;
		}
	});

	return numCorrect;
}

function write_file(matrix, filename){
	//  const d = matrix.map((row, i)=> row.map((col,j)=> matrix[j][i]));
	const str = matrix.reduce((prev, cur)=> prev + cur.reduce((p, c)=> p + c + "\t" , "") + "\n" ,"");
	fs.writeFileSync(filename, str);
}

module.exports = {
	read_file, 
	parse_data, 
	shuffle_arrays, 
	classification_error, 
	mat_transpose, 
	write_file
};
