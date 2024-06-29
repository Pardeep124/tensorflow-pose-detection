const data = {
  name: 'pardeep',
  change(name){
    this.name = name;
  }
}

const value  = data;

value.change('Pardeep Singh');

console.log(data);