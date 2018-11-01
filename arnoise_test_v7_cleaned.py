# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from numpy import pi, exp, dot
from matplotlib import pyplot as plt
from scipy.signal import freqz, lfilter
from ar_burg import arburg
from plot_zplane import zplane
from scipy.signal import correlate


#a0 = np.array([1, -0.5500, -0.1550, 0.5495, -0.6241])   
a0 = np.array([1, -0.8600, 1.0494, -0.6680, 0.9592, -0.7563, 0.5656])
#a0 = np.array([1, -2.7607, 3.8106, -2.6535, 0.9238])
#a0 = np.array([1, -2.7377, 3.7476, -2.6293, 0.9224])
#a0 = np.array([1, .6, -.2975, -.1927, .6329, .7057])
p0 = np.roots(a0)


a0=np.poly(p0)
#b0=np.poly(q0)
sv=1.0
dbSpec = 0.
N=4000

vn=np.random.randn(N)*sv
ss=lfilter([1.], a0, vn)
sw = np.sqrt(ss.var()/(10.**(dbSpec/10)))
wn=np.random.randn(N)*sw
sig=ss+wn
db_calc=10*np.log10(ss.var()/wn.var())

tf.reset_default_graph()


def get_pq(p_init, q_init):
    num_p, num_q = len(p_init), len(q_init)
    nin = tf.ones((1,30))
    b_init = np.concatenate([np.real(p_init), np.imag(p_init), 
                             np.real(q_init), np.imag(q_init)])
    k1 = tf.get_variable('k1', initializer=tf.zeros((30, 10)))
    b1 = tf.get_variable('b1', initializer=tf.zeros((10,)))
    h1 = nin@k1+b1#
    h1_a =  tf.nn.tanh(h1)
    k2 = tf.get_variable('k2', initializer=tf.zeros((10, 2*(num_p+num_q))))
    b2 = tf.get_variable('b2', initializer=tf.constant(b_init, dtype=tf.float32))
    h2 = h1_a@k2+b2
    ho = tf.tanh(tf.squeeze(h2))
    
    p_real, p_imag = ho[:num_p], ho[num_p:2*num_p]
    q_real, q_imag = ho[-2*num_q:-num_q], ho[-num_q:]

    p = tf.complex(p_real, p_imag)
    q = tf.complex(q_real, q_imag)
    return (p, q), [k1,b1,k2,b2]

def get_logtf(w_in, p, q):
    z_in = tf.exp(tf.complex(0., w_in))
    num_fact = tf.log(tf.abs(1-(z_in**-1)[:, None]@q[None, :]))
    den_fact = tf.log(tf.abs(1-(z_in**-1)[:, None]@p[None, :]))
    ltf = tf.reduce_sum(num_fact, axis=-1)-tf.reduce_sum(den_fact, axis=-1)
    return ltf


ae = arburg(sig, len(p0))[0]
pe = np.roots(ae)
(p,q) ,var_list = get_pq(pe, pe)
#(p,q) ,var_list = get_pq(np.zeros([len(p0)]), np.zeros([len(p0)]))
w_in = tf.placeholder(tf.float32, shape=(None,))
ltf_t = tf.placeholder(tf.float32, shape=(None,))
ltf = get_logtf(w_in, p, q)


ew_loss=(ltf_t-ltf)**4
m_loss = tf.reduce_mean(ew_loss)


r_loss = tf.Variable(0., trainable=False)
asgn_op = r_loss.assign(.99*r_loss+.01*m_loss)
with tf.control_dependencies([asgn_op]):
    bmask = ew_loss>r_loss

a_loss = tf.reduce_mean(tf.boolean_mask(ew_loss, bmask))

loss = m_loss+2*a_loss


lr = tf.Variable(1e-3, trainable=False)

sig_gn = tf.Variable(0.0005, trainable=False)
opt = tf.train.AdamOptimizer(lr)#, beta1=.999
gvs = opt.compute_gradients(loss)
cgvs = [(tf.clip_by_value(grad, -.01, .01), var) for grad, var in gvs]
add_noises = [var.assign(var+tf.random_normal(tf.shape(var), mean=0., stddev=sig_gn))
                                                        for _, var in cgvs]
train_op = opt.apply_gradients(cgvs)
with tf.control_dependencies(add_noises):    
    train_opn = opt.apply_gradients(cgvs)


steps = 100000
bsize = 512

def get_ltfvals(sig, nall=50000, rrange = np.r_[18:26]):
    wall = np.linspace(-pi, pi, nall)
    zall = exp(1j*wall)
    ltall = np.zeros_like(zall)
    
    for k in rrange:
        ae = arburg(sig, len(p0)+k)[0]    
        ltall += -np.log(np.polyval(np.flipud(ae), zall**-1))
    ltall /= len(rrange)
    return ltall, wall


def get_tf(aa, bb, N0):
    return np.fft.fftshift(np.log(np.fft.fft(bb,N0)/np.fft.fft(aa,N0)))
 

nall = 50000
ltfall, wall = get_ltfvals(sig, nall)
ltall = np.real(ltfall)

N0 = 2049
P0, w0= get_ltfvals(sig, N0)   

sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
sess.run(tf.global_variables_initializer())
avg_ls = 0.
lmda = .985

T = 2000

plt.close('all')
fig=plt.figure(figsize=(10,5))
ax1=plt.subplot(131)
zplane([1.], a0, ax=ax1)
ax2=plt.subplot(132)
ax2.plot(w0,np.real(P0), 'r--')
ax3=plt.subplot(133)
ax3.plot(w0,np.unwrap(np.imag(P0)), 'r--')
with sess.as_default():
    pe=p.eval()
    qe=q.eval()
    pp=ax1.scatter(np.real(pe), np.imag(pe), marker='x')
    qq=ax1.scatter(np.real(qe), np.imag(qe), marker='o')
    Pe = get_tf(np.poly(pe), np.poly(qe), N0)
    PP=ax2.plot(w0,np.real(Pe), 'b')+ax3.plot(w0,np.unwrap(np.imag(Pe)), 'b')
    bmp = ax2.scatter(0,0, marker='^')
    plt.tight_layout()
    plt.show()
    for k in range(steps):
        ns = np.random.randint(0, nall, size=(bsize,))
        ws = wall[ns]
        lts = ltall[ns]
        
        Tc = T*(2**np.floor(np.log(1+k/T)/np.log(2)))
        _, bm, ls = sess.run([train_opn,bmask,loss], 
                             feed_dict={w_in:ws, ltf_t:lts, 
                                        sig_gn:.0005*np.cos(2*pi*k/T)})
        avg_ls = lmda*avg_ls+(1-lmda)*ls
        if not (k+1)%100:
            print('step : {:5d} | loss = {:0.7f}'.format(k+1, avg_ls))
            pe=p.eval()
            qe=q.eval()
            pp.remove()
            qq.remove()
            bmp.remove()
            Pe = get_tf(np.poly(pe), np.poly(qe), N0)
            for el in PP: el.remove()
            PP=ax2.plot(w0,np.real(Pe), 'b')+ax3.plot(w0,np.unwrap(np.imag(Pe)), 'b')
            pp=ax1.scatter(np.real(pe), np.imag(pe), marker='x')
            qq=ax1.scatter(np.real(qe), np.imag(qe))
            bmp=ax2.scatter(ws[bm]/pi, np.nonzero(bm)[0]*0, marker='^')
            plt.pause(.01)


with sess.as_default():
    print(p.eval())
    print(q.eval())
    
    

