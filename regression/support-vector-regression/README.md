Main formulas required to understand SVM.py file:

regularizer term is min(||w||^2 /2)
utilized the maximize the margin and minimize loss

gradient descent needs the regularizer and the error term to define an optimization term:
- we intend on using a hinge loss function for SVM, which looks like this: <br>
<span>
f(x) = max{0, 1-t}
</span>

- t = y<sub>n</sub>(w<sup>T</sup>x + b)

- this, this is the following error term: <br>
<span>
(C<sub>i</sub>) * &sum;<sub>i=1</sub><sup>n</sup> (max{0, 1-y<sub>n</sub>(w<sup>T</sup>x + b)}) 
</span>

 - 