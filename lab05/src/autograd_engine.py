import math

# ==========================================
# Task 1: Value Node
# ==========================================
class Value:
    """
    A scalar node in a computation graph.

    Stores:
        data      : the scalar value computed in the forward pass
        grad      : dLoss/dThis, filled in during the backward pass
        _prev     : parent nodes (the inputs this node was computed from)
        _backward : local derivative rule, pushes this node's grad to its parents
        _op       : name of the operation that produced the node (for inspection)
    """
    def __init__(self, data, _children=(), _op=''):
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_children)
        self._backward = lambda: None
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data:.6f}, grad={self.grad:.6f}, op='{self._op}')"

    # ==========================================
    # Task 2: Operations (each one builds the graph during the forward pass)
    # ==========================================
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # d(a*b)/da = b, d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "Only constant int/float powers are supported"
        out = Value(self.data ** exponent, (self,), f'**{exponent}')

        def _backward():
            # d(a^n)/da = n * a^(n-1)
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), 'relu')

        def _backward():
            # d relu(a)/da = 1 if a > 0 else 0
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    # Extra operations needed for the sigmoid + BCE model in Task 5
    def sigmoid(self):
        x = self.data
        # Numerically stable sigmoid
        s = 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))
        out = Value(s, (self,), 'sigmoid')

        def _backward():
            # d sigmoid(a)/da = s * (1 - s)
            self.grad += s * (1.0 - s) * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), 'log')

        def _backward():
            # d log(a)/da = 1 / a
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    # Operations expressed through the primitives above
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __truediv__(self, other): return self * other ** -1
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __rsub__(self, other): return other + (-self)
    def __rtruediv__(self, other): return other * self ** -1

    # ==========================================
    # Task 3: Backward Pass
    # ==========================================
    def topological_sort(self):
        """Returns all nodes of the graph ordered so that parents come before children."""
        topo = []
        visited = set()
        # Iterative DFS (avoids Python's recursion limit on deep graphs)
        stack = [(self, False)]
        while stack:
            node, parents_done = stack.pop()
            if parents_done:
                topo.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            for parent in node._prev:
                if parent not in visited:
                    stack.append((parent, False))
        return topo

    def backward(self):
        # 1. Topologically sort the graph
        topo = self.topological_sort()

        # 2. Set output gradient to 1 (dOut/dOut = 1)
        self.grad = 1.0

        # 3. Visit nodes in reverse order
        for node in reversed(topo):
            # 4. Each local rule uses += so gradients accumulate, not overwrite
            node._backward()


# ==========================================
# Sanity Checks
# ==========================================
if __name__ == "__main__":
    print("Testing Value node, operations and backward pass\n")

    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    out = ((a * b + c) ** 2).relu()

    # 1. Graph construction during the forward pass
    topo = out.topological_sort()
    print(f"1. Forward result: {out.data} (Expected: 16.0)")
    print(f"2. Nodes in graph: {len(topo)} | ops in order: {[n._op for n in topo if n._op]}")
    assert out.data == 16.0, "Forward check failed!"
    assert out._prev[0]._op == '**2', "Parent check failed!"

    # 2. Backward pass: d/da = 2(ab+c)*b = -24, d/db = 2(ab+c)*a = 16, d/dc = 2(ab+c) = 8
    out.backward()
    print(f"3. Gradients: a={a.grad}, b={b.grad}, c={c.grad} (Expected: -24.0, 16.0, 8.0)")
    assert (a.grad, b.grad, c.grad) == (-24.0, 16.0, 8.0), "Gradient check failed!"

    # 3. Accumulation: x is used twice, y = x*x + x -> dy/dx = 2x + 1 = 7
    x = Value(3.0)
    y = x * x + x
    y.backward()
    print(f"4. Accumulated gradient for reused node: {x.grad} (Expected: 7.0)")
    assert x.grad == 7.0, "Accumulation check failed!"

    print("\n✅ All autograd engine sanity checks passed!")
