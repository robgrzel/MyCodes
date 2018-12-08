################################################################################
################################ COMMON.PY #####################################
################################################################################

def ipython_info():
	import sys
	ip = False
	if 'ipykernel' in sys.modules:
		ip = True  # 'notebook'
	# elif 'IPython' in sys.modules:
	#    ip = 'terminal'
	return ip


ISIPYTHON = ipython_info()
DEBUG = 1

if not ISIPYTHON:
	print('working in file: ', __file__)
	import matplotlib

	# matplotlib.use('tkagg')
	matplotlib.use('agg')

else:
	print ('script is working in ipython/jupyter')

import time
import json
import traceback
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pandas as pd

from pprint import pprint as pprint

np.random.seed(0)

import os

for dir in ['figs', 'logs', 'data']:
	try:
		os.mkdir(dir)
	except:
		pass


def best_parameters(W, Theta, HL):
	best = {}
	best['W'] = W.copy()
	best['Theta'] = Theta.copy()
	best['C'] = dict(Tr=1e6, Va=1e6, Te=1e6)
	return best


def init_C_H_U(epochs, L):
	C = {}
	H = {}
	U = {'Tr': {}, 'Va': {}, 'Te': {}, }
	C['Tr'] = np.zeros([epochs + 1])
	C['Va'] = np.zeros([epochs + 1])
	C['Te'] = np.zeros([epochs + 1])
	H['Tr'] = np.zeros([epochs + 1])
	H['Va'] = np.zeros([epochs + 1])
	H['Te'] = np.zeros([epochs + 1])
	for l in range(1, L):
		U['Tr'][l] = np.zeros([epochs + 1])
		U['Te'][l] = np.zeros([epochs + 1])
		U['Va'][l] = np.zeros([epochs + 1])
	return C, H, U


def init_C(epochs):
	C = {}
	C['Tr'] = np.zeros([epochs + 1])
	C['Va'] = np.zeros([epochs + 1])
	C['Te'] = np.zeros([epochs + 1])
	return C


def timefunc(name, func, *args, **kwargs):
	start = time.time()
	out = func(*args, **kwargs)
	end = time.time()

	print('executing %s took: %f' % (name, end - start))
	return out


def ppr(x, lvl=-2):
	stack = traceback.extract_stack()
	filename, lineno, funcname, code = stack[lvl]
	print('%s' % code)
	pprint(x)


def pprsh(x):
	if type(x) == dict:
		x = {k: c.shape for k, c in x.items()}
	elif type(x) == list:
		x = [c.shape for c in x]
	elif type(x) == type(np.array(0)):
		x = x.shape
	ppr(x, lvl=-3)


def training_containers(shapeNN, mB):
	L = len(shapeNN)
	hL = L - 1

	b = {}
	W = {}
	V = {}
	dV = {}
	dW = {}
	Delta = {}
	Theta = {}
	dTheta = {}

	for l in range(1, L):
		N = shapeNN[l - 1]
		M = shapeNN[l]

		b[l] = np.zeros([M, mB], dtype=float)
		W[l] = np.zeros([M, N], dtype=float)
		dW[l] = np.zeros([M, N], dtype=float)
		dV[l] = np.zeros([M, mB], dtype=float)
		V[l] = np.zeros([M, mB], dtype=float)
		Delta[l] = np.zeros([M, mB], dtype=float)
		Theta[l] = np.zeros([M, 1], dtype=float)
		dTheta[l] = np.zeros([M, 1], dtype=float)

	N = shapeNN[0]
	M = shapeNN[-1]
	V[0] = np.zeros([N, mB], dtype=float)
	V['Z'] = V[L - 1].copy()

	return W, Theta, dW, dTheta, V, dV, b, Delta


def validation_containers(shapeNN, p):
	L = len(shapeNN)
	hL = L - 1

	b = {}
	V = {}

	for l in range(1, L):
		N = shapeNN[l - 1]
		M = shapeNN[l]

		b[l] = np.zeros([M, p], dtype=float)
		V[l] = np.zeros([M, p], dtype=float)

	return b, V


def validation_full_containers(shapeNN, p):
	L = len(shapeNN)
	hL = L - 1

	b = {}
	V = {}
	dV = {}
	Delta = {}
	dTheta = {}

	for l in range(1, L):
		N = shapeNN[l - 1]
		M = shapeNN[l]

		b[l] = np.zeros([M, p], dtype=float)
		V[l] = np.zeros([M, p], dtype=float)
		dV[l] = np.zeros([M, p], dtype=float)
		Delta[l] = np.zeros([M, p], dtype=float)
		dTheta[l] = np.zeros([M, 1], dtype=float)

	return b, V, dV, Delta, dTheta


def init_weights(trials, M, N, targetType='0/1'):
	if trials == 0:
		shapeNN = (M, N)
	else:
		shapeNN = (trials, M, N,)

	#variance has to be 1/N, variance = stdv^2, so:
	var = 1. / N
	std = var ** 0.5
	initWji = (np.random.normal(loc=0, scale=std, size=shapeNN))

	if targetType == '-1/1':
		initWji *= 2.
		initWji -= 1.
	initWji = np.require(initWji, requirements='C')
	return initWji


def init_parameters(shapeNN, Theta, W):
	L = len(shapeNN)
	hL = L - 1

	inits = {}

	for l in range(1, L):
		N = shapeNN[l - 1]
		M = shapeNN[l]

		Theta[l][:M] = np.zeros([M, 1], dtype=float)
		W[l][:M, :N] = init_weights(0, M, N, )

	inits['Theta'] = Theta.copy()
	inits['W'] = W.copy()

	return inits


def generate_mini_batches(epochs, iterations, mB, pTr):
	muIdx = np.zeros([epochs, iterations, mB], dtype=int)

	for epoch in range(epochs):
		muIdx[epoch].flat = np.random.permutation(np.arange(pTr))

	return muIdx


def sigmoid(b, out=None):
	if out is None:
		return 1. / (1. + np.exp(-1. * b))
	else:
		out[:] = 1. / (1. + np.exp(-1. * b))


def dsigmoid(sigmoid, out=None):
	if out is None:
		return sigmoid * (1.0 - sigmoid)
	else:
		out[:] = sigmoid * (1.0 - sigmoid)


def cerror(C, O, t, epoch, p, M):
	C[epoch] = 1. / float(2. * p) * np.sum(np.abs(t[:M, :p] - O[:M, :p]))


def energy(H, O, t, epoch, p, M):
	H[epoch] = 0.5 * np.sum(np.power(t[:M, :p] - O[:M, :p], 2.))


def accuracy(x):
	return (1. - x) * 100.


def learning_speed(U, dTheta, epoch, L):
	for l in range(1, L):
		U[l][epoch] = np.linalg.norm(dTheta[l])


def validate_full(g, dg, Zi, W, Theta, b, V, dV, Delta, dTheta, C, U, H, epoch, shapeNN, p):
	L = len(shapeNN)
	HL = L - 1

	for l in range(1, L):
		N = shapeNN[l - 1]
		M = shapeNN[l]
		b[l][:M, :p] = np.einsum('ij,jm->im', W[l][:M, :N], V[l - 1][:N, :p]) - Theta[l][:M]
		V[l][:M, :p] = g(b[l][:M, :p])
		dV[l][:M, :p] = dg(V[l][:M, :p])

	M = shapeNN[-1]
	Delta[HL][:M, :p] = (Zi[:M, :p] - V[HL][:M, :p]) * dV[HL][:M, :p]

	for l in reversed(range(1, HL)):
		N = shapeNN[l]
		M = shapeNN[l + 1]
		Delta[l][:N, :p] = np.einsum('im,ij->jm', Delta[l + 1][:M, :p], W[l + 1][:M, :N]) * dV[l][:N, :p]

	for l in range(1, L):
		M = shapeNN[l]
		dTheta[l][:M, 0] = np.sum(Delta[l][:M, :p], axis=1)

	cerror(C, V[L - 1], Zi, epoch, p, shapeNN[-1])
	energy(H, V[L - 1], Zi, epoch, p, shapeNN[-1])
	learning_speed(U, dTheta, epoch, L)


# g, xTrainNorm, tTrain, W, Theta, bTr, VTr, C['Tr'], epoch, shapeNN, pTr
def validate(g, Zi, W, Theta, b, V, C, epoch, shapeNN, p):
	L = len(shapeNN)

	for l in range(1, L):
		N = shapeNN[l - 1]
		M = shapeNN[l]
		b[l][:M, :p] = np.einsum('ij,jm->im', W[l][:M, :N], V[l - 1][:N, :p]) - Theta[l][:M]
		V[l][:M, :p] = g(b[l][:M, :p])

	cerror(C, V[L - 1], Zi, epoch, p, shapeNN[-1])


def plot_all_C(taskC, task):
	"""
	Plot the classification error of the training set and of the validation set
	 as a function of epoch for the four networks specified above.
	Use logarithmical scaling of the yy-axis for the plot.
	:return:
	"""
	fig = plt.figure(figsize=(15, 10))
	plt.title('FFR135 HW %s : Classification error in logscale' % task)

	ax = plt.subplot(111)
	color = iter(cm.rainbow(np.linspace(0, 1, 2 * len(taskC.keys()))))
	maxC = 0
	minC = 1e6
	for shapeNN in taskC.keys():
		C = taskC[shapeNN]
		CTr = C['Tr']
		CVa = C['Va']
		c1 = next(color)
		c2 = next(color)
		ax.plot(CTr, ':', c=c1, zorder=1, lw=3, label='C(Tr) %s' % (shapeNN,))
		ax.plot(CVa, '-', c=c2, zorder=2, lw=2, label='C(Va) %s' % (shapeNN,))
		ax.plot(0, 0)
		ax.plot(30, 0)
		ax.plot(0, 50)
		maxC = max(np.max(CVa), max(np.max(CTr), maxC))
		minC = min(np.min(CVa), min(np.min(CTr), minC))

	# ax.plot(CTe, c='g', zorder=3, lw=1, label='CTe')
	plt.grid(True)
	plt.legend()
	ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200])

	ax.set_yscale('log')
	ax.set_ylim(minC, maxC)

	ax.set_xlabel('epoch [-]')
	ax.set_ylabel('log C [%]')

	plt.savefig("figs/FFR135_HW%s_C.png" % task)

	if ISIPYTHON:
		plt.show()


def plot_all_U(taskU, task):

	fig = plt.figure(figsize=(15, 10))
	plt.title('FFR135 HW %s : Learning speed U for l-layer' % task)
	ax = plt.subplot(111)

	nlines = 0
	for shapeNN in taskU.keys():
		layers = taskU[shapeNN].keys()
		nlines += len(layers)

	color = iter(cm.rainbow(np.linspace(0, 1, nlines)))
	maxUTr = 0
	minUTr = 1e6
	for shapeNN in taskU.keys():
		layers = taskU[shapeNN].keys()
		for l in layers:
			UTr = taskU[shapeNN][l]
			c1 = next(color)
			ax.plot(UTr, ':', c=c1, zorder=1, lw=3, label='U(Tr) L:%d, NN:%s' % (l, shapeNN,))
			maxUTr = max(np.max(UTr), maxUTr)
			minUTr = min(np.min(UTr), minUTr)
			ax.plot(0, maxUTr)

	plt.grid(True)
	plt.legend()
	ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200, int(maxUTr)])

	ax.set_yscale('log')
	ax.set_ylim(minUTr, maxUTr * 1.05)

	ax.set_xlabel('epoch [-]')
	ax.set_ylabel('log U(l) [%]')

	plt.savefig("figs/FFR135_HW%s_U.png" % task)

	if ISIPYTHON:
		plt.show()


def plot_all_H(taskH, task):

	fig = plt.figure(figsize=(15, 10))
	plt.title('FFR135 HW %s : Energy Functions H' % task)

	ax = plt.subplot(111)
	color = iter(cm.rainbow(np.linspace(0, 1, len(taskH.keys()))))
	maxHTr = 0
	minHTr = 1e6
	for shapeNN in taskH.keys():
		H = taskH[shapeNN]
		HTr = H['Tr']
		maxHTr = max(np.max(HTr), maxHTr)
		minHTr = min(np.min(HTr), minHTr)
		c1 = next(color)
		ax.plot(HTr, ':', c=c1, zorder=1, lw=3, label='H(Tr) %s' % (shapeNN,))
		ax.plot(0, 0)
		ax.plot(30, 0)

	plt.grid(True)
	plt.legend()
	ax.set_ylim(minHTr, maxHTr * 1.05)

	ax.set_xlabel('epoch [-]')
	ax.set_ylabel('H')

	plt.savefig("figs/FFR135_HW%s_H.png" % task)

	if ISIPYTHON:
		plt.show()


def plot_activation(mB, t, epoch, task, b, V, dV, shapeNN, L):
	if t == 0 and epoch == 29:
		fig = plt.figure()
		plt.title('Task:%s, activation at epoch[%d]' % (task, epoch))

	for l in range(1, L):
		N = shapeNN[l - 1]
		M = shapeNN[l]

		if t == 0 and epoch == 29:
			plt.plot(np.mean(b[l][:M, :mB], axis=1), np.mean(V[l][:M, :mB], axis=1), '.',
			         label='V[l=%d][M=%d][task:%s]' % (l, M, task))
			plt.plot(np.mean(b[l][:M, :mB], axis=1), np.mean(dV[l][:M, :mB], axis=1), '.',
			         label='dV[l=%d][M=%d][task:%s]' % (l, M, task))

	if t == 0 and epoch == 29:
		plt.legend()
		plt.savefig('figs/ACTIVATION(EPOCH=%d)_HW%s.png' % (epoch, task))


def plot_loss_accuracy(train_losses, train_acc, test_losses, test_acc,plot_title="Loss, train acc, test acc",step=100):
    """
    Function generates matplolib plots with loss and accuracies
    
    
    Parameters
    ----------
    losses : list
        list with values of loss function, computed every 'step' trainning iterations
    train_acc : list
        list with values  of training acuracies, computed every 'step' trainning iterations
    test_acc : list
        list with values  of testing acuracies, computed every 'step' trainning iterations
    step : int
        number of trainning iteration after which we compute (loss and accuracies)
    plot_title: string
        title of the plot
    
    Raises
    ------
    Exception
        when an error occure
    """
        
    training_iters = len(train_losses)
    # iters_steps
    iter_steps = [step *k for k in range(training_iters)]

    imh = plt.figure(1, figsize=(15, 14), dpi=160)
    # imh.tight_layout()
    # imh.subplots_adjust(top=0.88)

    final_acc = test_acc[-1]
    img_title = "{}, test acc={:.4f}".format(plot_title,final_acc)
    imh.suptitle(img_title)
    plt.subplot(221)
    #plt.plot(iter_steps,losses, '-g', label='Loss')
    plt.semilogy(iter_steps, train_losses, '-g', label='Trn Loss')
    plt.title('Train Loss ')
    plt.subplot(222)
    plt.plot(iter_steps, train_acc, '-r', label='Trn Acc')
    plt.title('Train Accuracy')

    plt.subplot(223)
    plt.semilogy(iter_steps, test_losses, '-g', label='Tst Loss')
    plt.title('Test Loss')
    plt.subplot(224)
    plt.plot(iter_steps, test_acc, '-r', label='Tst Acc')
    plt.title('Test Accuracy')


    #plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    plot_file = "./plots/{}.png".format(plot_title.replace(" ","_"))
    plt.savefig(plot_file)
    plt.show()
        
# TODO

def plot_recognized(xTrainNorm, tTrain, xValidNorm, tValid, xTestNorm):
	# TODO
	return
	nSamples = 20

	fig, ax = plt.subplots(nSamples, nSamples, sharex='all', sharey='all', squeeze=True, clear=True,
	                       gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(nSamples / 2, nSamples / 2))

	ax = ax.ravel()
	for mi in range(nSamples * nSamples):
		ax[mi].imshow(xTrainNorm[:, mi].reshape([28, 28]), cmap='gray')
		isEqual = np.all(np.argmax(OiTr[:, mi]) == np.argmax(tTrain[:, mi]))
		if isEqual == 0:
			ax[mi].text(0.5, 25, np.argmax(tTrain[:, mi]), color='r', fontsize=15)
			ax[mi].text(20, 25, np.argmax(OiTr[:, mi]), color='b', fontsize=15)
		ax[mi].set_xticklabels([])
		ax[mi].set_yticklabels([])
		ax[mi].set_aspect('equal')
	plt.subplots_adjust(wspace=0, hspace=0)
	plt.savefig("figs/FFR135_HW3.1.3_TRAIN.png")

	fig, ax = plt.subplots(nSamples, nSamples, sharex='all', sharey='all', squeeze=True, clear=True,
	                       gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(nSamples / 2, nSamples / 2))

	ax = ax.ravel()
	for mi in range(nSamples * nSamples):
		ax[mi].imshow(xValidNorm[:, mi].reshape([28, 28]), cmap='gray')
		isEqual = np.all(np.argmax(OiVa[:, mi]) == np.argmax(tValid[:, mi]))
		if isEqual == 0:
			ax[mi].text(0.5, 25, np.argmax(tValid[:, mi]), color='r', fontsize=15)
			ax[mi].text(20, 25, np.argmax(OiVa[:, mi]), color='b', fontsize=15)
		ax[mi].set_xticklabels([])
		ax[mi].set_yticklabels([])
		ax[mi].set_aspect('equal')
	plt.subplots_adjust(wspace=0, hspace=0)
	plt.savefig("figs/FFR135_HW3.1.3_VALID.png")

	fig, ax = plt.subplots(nSamples, nSamples, sharex='all', sharey='all', squeeze=True, clear=True,
	                       gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(nSamples / 2, nSamples / 2))

	ax = ax.ravel()
	for mi in range(nSamples * nSamples):
		ax[mi].imshow(xTestNorm[:, mi].reshape([28, 28]), cmap='gray')
		isEqual = np.all(np.argmax(OiTe[:, mi]) == np.argmax(tTest[:, mi]))
		if isEqual == 0:
			ax[mi].text(0.5, 25, np.argmax(tTest[:, mi]), color='r', fontsize=15)
			ax[mi].text(20, 25, np.argmax(OiTe[:, mi]), color='b', fontsize=15)
		ax[mi].set_xticklabels([])
		ax[mi].set_yticklabels([])
		ax[mi].set_aspect('equal')

	plt.subplots_adjust(wspace=0, hspace=0)
	plt.savefig("figs/FFR135_HW3.1.3_TEST.png")
