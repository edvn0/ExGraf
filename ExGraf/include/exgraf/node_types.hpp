#pragma once

namespace ExGraf {

#define EXGRAF_NODE_LIST_FORWARD                                               \
	X(Node)                                                                      \
	X(Placeholder)                                                               \
	X(Variable)                                                                  \
	X(Add)                                                                       \
	X(Mult)                                                                      \
	X(ReLU)                                                                      \
	X(Tanh)                                                                      \
	X(Softmax)                                                                   \
	X(CrossEntropyLoss)                                                          \
	X(SumAxis)                                                                   \
	X(Log)                                                                       \
	X(Neg)                                                                       \
	X(MSELoss)                                                                   \
	X(Subtract)                                                                  \
	X(Hadamard)

// Templated version that uses the base list
#define EXGRAF_NODE_LIST(T)                                                    \
	X(Node<T>)                                                                   \
	X(Placeholder<T>)                                                            \
	X(Variable<T>)                                                               \
	X(Add<T>)                                                                    \
	X(Mult<T>)                                                                   \
	X(ReLU<T>)                                                                   \
	X(Tanh<T>)                                                                   \
	X(Softmax<T>)                                                                \
	X(CrossEntropyLoss<T>)                                                       \
	X(SumAxis<T>)                                                                \
	X(Log<T>)                                                                    \
	X(Neg<T>)                                                                    \
	X(MSELoss<T>)                                                                \
	X(Subtract<T>)                                                               \
	X(Hadamard<T>)

} // namespace ExGraf
