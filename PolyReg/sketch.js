let x_points = [];
let y_points = [];
const lineX = [];


let m1;
let m2;
let m3;
let b;
let optimizer;

let mode = 2;

function predict(arr) {
   const xs = tf.tensor1d(arr);
   if (mode == 0) {
      const ys = xs.mul(m3).add(b);
      return ys;
   }
   if (mode == 1) {
      const x2 = xs.square().mul(m2);
      const x = xs.mul(m3);
      const ys = x2.add(x).add(b);
      return ys;
   }
   if (mode == 2) {
      const h = tf.scalar(3);
      const x3 = xs.pow(h).mul(m1);
      const x2 = xs.square().mul(m2);
      const x = xs.mul(m3);
      const ys = x3.add(x2).add(x).add(b);
      return ys;
   }
}

function loss(pred, labels) {
   return pred.sub(labels).square().mean();
}

function setup() {
   createCanvas(600, 600);
   const lm1 = tf.scalar(random(-1, 1));
   const lm2 = tf.scalar(random(-1, 1));
   const lm3 = tf.scalar(random(-1, 1));
   const lb = tf.scalar(random(-1, 1));
   m1 = lm1.variable();
   m2 = lm2.variable();
   m3 = lm3.variable();
   b = lb.variable();
   optimizer = tf.train.adam(0.1);
   lm1.dispose();
   lm2.dispose();
   lm3.dispose();
   lb.dispose();
   let start = -1;
   for (let i = 0; i < 1000; i++) {
      lineX.push(start);
      start += 1 / 500;
   }
   let LR = select('#LinearRegression');
   LR.mousePressed(() => {mode = 0});
   let QR = select('#QuadraticRegression');
   QR.mousePressed(() => {mode = 1});
   let CR = select('#cubicRegression');
   CR.mousePressed(() => {mode = 2});
}

function draw() {
   background(0);
   stroke(255);
   strokeWeight(8);
   if (x_points.length > 0) {
      tf.tidy(() => {
         const ys = tf.tensor1d(y_points);
         optimizer.minimize(() => loss(predict(x_points), ys));
      });
   }
   for (let i = 0; i < x_points.length; i++) {
      const x = map(x_points[i], -1, 1, 0, width);
      const y = map(y_points[i], -1, 1, height, 0);
      point(x, y);
   }
   const lineYt = tf.tidy(() => predict(lineX));
   const lineY = tf.tidy(() => lineYt.arraySync());
   drawLines(lineX, lineY);
   lineYt.dispose();
}

function drawLines(lineX, lineY) {
   strokeWeight(2);
   noFill();
   beginShape();
   for (let i = 0; i < lineX.length; i++) {
      const x = map(lineX[i], -1, 1, 0, width);
      const y = map(lineY[i], -1, 1, height, 0);
      vertex(x, y);
   }
   endShape();
}

function mousePressed() {
   if (mouseX > 0 && mouseX < width && mouseY > 0 && mouseY < height) {
      x_points.push(map(mouseX, 0, width, -1, 1));
      y_points.push(map(mouseY, 0, height, 1, -1));
   }
}