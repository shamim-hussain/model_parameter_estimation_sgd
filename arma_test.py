# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from numpy import pi, exp, dot
from matplotlib import pyplot as plt
from scipy.signal import freqz, lfilter
from ar_burg import arburg
from plot_zplane import zplane
from spectrum import arma_estimate


a0 = np.array([1, -0.5500, -0.1550, 0.5495, -0.6241])   
#a0 = np.array([1, -0.8600, 1.0494, -0.6680, 0.9592, -0.7563, 0.5656])
#a0 = np.array([1, -2.7607, 3.8106, -2.6535, 0.9238])
#a0 = np.array([1, -2.7377, 3.7476, -2.6293, 0.9224])
#a0 = np.array([1, .6, -.2975, -.1927, .6329, .7057])
p0 = np.roots(a0)
#p0=np.array([.9*exp(1j*2*pi/3), .9*exp(-1j*2*pi/3),
#             .85*exp(1j*1*pi/5), .85*exp(-1j*1*pi/5), -.8, .9])
#q0=np.array([.9*exp(1j*5*pi/6), .9*exp(-1j*5*pi/6), .8*exp(1j*pi/4), .8*exp(-1j*pi/4)])

#p0=[.8*exp(1i*11*pi/13), .8*exp(-1i*11*pi/13),.86*exp(1i*3*pi/7), .86*exp(-1i*3*pi/7), .7].'; %,,, .9 
#p0=[.95*exp(1i*5*pi/13), .95*exp(-1i*5*pi/13),.95*exp(1i*6*pi/13), .95*exp(-1i*6*pi/13), .98*exp(1i*7*pi/13), .98*exp(-1i*7*pi/13)].';%, .9  %Burg best

#b0 = np.array([1, -0.5500, -0.1550, 0.5495, -0.6241])   
b0 = np.array([1, -0.8600, 1.0494, -0.6680, 0.9592, -0.7563, 0.5656])
#b0 = np.array([1, -2.7607, 3.8106, -2.6535, 0.9238])
#b0 = np.array([1, -2.7377, 3.7476, -2.6293, 0.9224])
#b0 = np.array([1, .6, -.2975, -.1927, .6329, .7057])
q0 = np.roots(b0)

num_p, num_q = len(p0), len(q0)
a0=np.poly(p0)
b0=np.poly(q0)
sv=1.0
sw=3.
N=600
vn=np.random.randn(N)*sv
wn=np.random.randn(N)*sw
s=lfilter(b0, a0, vn)
x=s#+wn
ae = arburg(x, len(p0)+20)[0]
ainit = arburg(x, len(p0))[0]
ainit = np.r_[[1.],arma_estimate(x, len(p0), len(p0), 60)[0][:len(p0)]]
pinit = np.roots(ainit)
hdim = num_p*10
#pinit = np.array([.9+.1j,.9-.1j,.9+.2j,.9-.2j, 0])+0j

tf.reset_default_graph()

z_in = tf.placeholder(tf.complex64, shape=(None,))

nin = tf.ones((1,30))
#nin = tf.get_variable('nin',initializer=tf.ones((1,30))*100)/100
k1 = tf.get_variable('k1', initializer=tf.zeros((30, 10)))
b1 = tf.get_variable('b1', initializer=tf.zeros((10,)))
h1 = nin@k1+b1#
#h1 = tf.get_variable('h1', initializer=tf.zeros((1,10)))
h1_a =  tf.nn.tanh(h1)#h1
#h1_a = h1/(tf.norm(h1)+1e-4)
k2 = tf.get_variable('k2', initializer=tf.zeros((10, 2*(num_p+num_q))))
b2 = tf.get_variable('b2', initializer=tf.zeros((2*(num_p+num_q),)))
h2 = h1_a@k2+b2
ho = tf.squeeze(h2)
ho = tf.nn.tanh(ho)

p_real = ho[:num_p]
p_imag = ho[num_p:2*num_p]

q_real = ho[-2*num_q:-num_q]
q_imag = ho[-num_q:]

p = tf.complex(p_real, p_imag)
q = tf.complex(q_real, q_imag)


num_fact = tf.log(1-(z_in**-1)[:, None]@q[None, :])
den_fact = tf.log(1-(z_in**-1)[:, None]@p[None, :])
ltf = tf.reduce_sum(num_fact, axis=-1)-\
        tf.reduce_sum(den_fact, axis=-1)#+cm

#pv = tf.placeholder(tf.complex64,p0.shape)
#pr_asgn = p_real.assign(tf.real(pv))
#pi_asgn = p_imag.assign(tf.imag(pv))


ltf_t = tf.placeholder(tf.complex64, shape=(None,))
ltf_tr = tf.real(ltf_t)
ltf_ti = tf.imag(ltf_t)
ltfr = tf.real(ltf)
ltfi = tf.imag(ltf)

#loss = tf.reduce_mean((den_t-den)**2)
se_loss=(ltf_tr-ltfr)**4
mse_loss = tf.reduce_mean(se_loss)

angle_diff = tf.abs(ltf_ti-ltfi)
#angle_loss = tf.reduce_mean(tf.minimum(angle_diff, 2*pi-angle_diff))/pi
angle_loss = tf.reduce_mean(tf.minimum(angle_diff, 2*pi-angle_diff)**2)/(pi**2)


#
#def get_median(v):
#  #v = tf.reshape(v, [-1])
#  m = tf.shape(v)[0]//2
#  return tf.nn.top_k(v, m).values[m-1]

smse_loss = tf.Variable(0., trainable=False)
asmse = smse_loss.assign(.99*smse_loss+.01*mse_loss)
with tf.control_dependencies([asmse]):
    bmask = se_loss>smse_loss
#bmask = se_loss>mse_loss
#targ = tf.boolean_mask(ltf_tr, bmask)
#val = tf.boolean_mask(ltfr, bmask)
tmse_loss = tf.reduce_mean(tf.boolean_mask(se_loss, bmask))
#tmse_loss = tf.reduce_mean(se_loss*tf.nn.sigmoid(se_loss-mse_loss))
targ = ltf_tr
val = ltfr
c_targ = targ-tf.reduce_mean(targ)
c_val = val-tf.reduce_mean(val)
corr_loss = 1.-tf.reduce_sum(c_targ*c_val)/ \
                (tf.norm(c_targ)*tf.norm(c_val)+1e-4)

et=tf.exp(ltf_t)
ev=tf.exp(ltf)
c_targ = et-tf.reduce_mean(et)
c_val = ev-tf.reduce_mean(ev)
ecorr_loss = 1.-tf.reduce_sum(c_targ*c_val)/ \
                (tf.norm(c_targ)*tf.norm(c_val)+1e-4)
emse_loss = tf.reduce_mean(tf.abs(et-ev)**2)
#wtm, wtc, wta = tf.Variable(1., trainable=False), tf.Variable(1., trainable=False), tf.Variable(1., trainable=False)
#loss = wtc*corr_loss+wtm*mse_loss+wta*angle_loss#+emse_loss#+ecorr_loss
loss = mse_loss+2*tmse_loss#+corr_loss#++wta*angle_loss#+emse_loss#+ecorr_loss

lr = tf.Variable(1e-3, trainable=False)
lrp=tf.placeholder(tf.float32, ())
alr=lr.assign(lrp)


sig_gn = tf.Variable(0.0005, trainable=False)
opt = tf.train.AdamOptimizer(lr)#, beta1=.999
gvs = opt.compute_gradients(loss)
#cgvs = [(tf.clip_by_value(grad, -.05, .05), var) for grad, var in gvs]
cgvs = [(tf.clip_by_value(grad, -.01, .01), var) for grad, var in gvs]
#cgvs = [(grad+tf.random_normal(tf.shape(grad), mean=0., stddev=sig_gn), var) 
#                                                        for grad, var in cgvs]
#sdict={}
#sgvs = []
#for grad, var in cgvs:
#    switch = tf.Variable(1., trainable=False)
#    sgvs.append((grad*switch, var))
#    sdict[var]=switch

add_noises = [var.assign(var+tf.random_normal(tf.shape(var), mean=0., stddev=sig_gn))
                                                        for _, var in cgvs]
train_op = opt.apply_gradients(cgvs)
with tf.control_dependencies(add_noises):    
    train_opn = opt.apply_gradients(cgvs)


steps = 100000
bsize = 512

from scipy.signal import correlate
N=len(x)
Rx = correlate(x, x)[N-50:50-N]/N
n=np.r_[-49:50]

sess = tf.Session()
sess.run(tf.global_variables_initializer())
avg_ls = 0.
lmda = .985

nall = 50000
rall = np.sqrt(np.random.uniform(1.**2, 1.**2, (nall,)))
wall = np.random.uniform( -pi, pi, (nall,))
zall = rall*exp(1j*wall)
ltall = np.zeros_like(zall)
rrange = np.r_[18:26]
for k in rrange:
    ae = arburg(x, len(p0)+k)[0]    
    ltall += -np.log(np.polyval(np.flipud(ae), zall**-1))
ltall /= len(rrange)

N0=2048
w0=2*np.arange(-((N0-1)//2),N0//2+1.)/N0
#P0 = 0
#for k in rrange:
#    ae = arburg(x, len(p0)+k)[0]
#    P0+=np.log(np.abs(1./np.fft.fftshift(np.fft.fft(ae,N0))))  
#P0 /= len(rrange)
P0 = []
for k in rrange:
    ae = arburg(x, len(p0)+k)[0]
    P0.append(np.log(1./np.fft.fftshift(np.fft.fft(ae,N0))))
P0 = np.stack(P0)
P0 = np.mean(P0, axis=0)
T = 2000

plt.close('all')
fig=plt.figure(figsize=(10,5))
ax1=plt.subplot(131)
zplane([1.], a0, ax=ax1)
ax1.scatter(np.real(q0), np.imag(q0), marker='^')
ax2=plt.subplot(132)
ax2.plot(w0,np.real(P0), 'r--')
ax3=plt.subplot(133)
ax3.plot(w0,np.unwrap(np.imag(P0)), 'r--')
with sess.as_default():
    pe=p.eval()
    qe=q.eval()
    pp=ax1.scatter(np.real(pe), np.imag(pe), marker='x')
    qq=ax1.scatter(np.real(qe), np.imag(qe), marker='o')
    bb=np.poly(qe)
    aa=np.poly(pe)
    Pe=np.log(np.fft.fftshift(np.fft.fft(bb,N0))/
                     np.fft.fftshift(np.fft.fft(aa,N0)))
    PP=ax2.plot(w0,np.real(Pe), 'b')+ax3.plot(w0,np.unwrap(np.imag(Pe)), 'b')
    bmp = ax2.scatter(0,0, marker='^')
    print(pe)
    print(qe)
    plt.tight_layout()
    plt.show()
    for k in range(steps):
        ns = np.random.randint(0, nall, size=(bsize,))
        zs = zall[ns]
        lts = ltall[ns]
        
        #dfs = np.log(np.abs(1-(zs**-1)[:,None]@p0[None,:]))
        #ds = np.sum(dfs, axis=-1)
        
        #ds = -np.log( np.abs((zs[:,None]**(-n))@Rx))/2.
        Tc = T*(2**np.floor(np.log(1+k/T)/np.log(2)))
        _, bm, ls = sess.run([train_opn,bmask,loss], feed_dict={z_in:zs, ltf_t:lts, sig_gn:.0005*np.cos(2*pi*k/T)})
                                                 #wtm:(1.+np.cos(2*pi*(k/T+.5)))/2,
                                                 #wta:(1.+np.cos(2*pi*k/T))/2})
        avg_ls = lmda*avg_ls+(1-lmda)*ls
        if not (k+1)%100:
            print('step : {:5d} | loss = {:0.7f}'.format(k+1, avg_ls))
            pe=p.eval()
            qe=q.eval()
            pp.remove()
            qq.remove()
            bmp.remove()
            bb=np.poly(qe)
            aa=np.poly(pe)
            Pe=np.log(np.fft.fftshift(np.fft.fft(bb,N0))/
                             np.fft.fftshift(np.fft.fft(aa,N0)))
            for x in PP: x.remove()
            PP=ax2.plot(w0,np.real(Pe), 'b')+ax3.plot(w0,np.unwrap(np.imag(Pe)), 'b')
            pp=ax1.scatter(np.real(pe), np.imag(pe), marker='x')
            qq=ax1.scatter(np.real(qe), np.imag(qe))
            bmp=ax2.scatter(np.angle(zs[bm])/pi, np.nonzero(bm)[0]*0, marker='^')
            plt.pause(.01)


with sess.as_default():
    print(p.eval())
    print(q.eval())
    
    
#alr.eval({lrp:1e-5}, sess)

