import numpy as np

M,K,N=16,64,16

A=np.random.rand(M,K)
B=np.random.rand(K,N)
# r

Afp16=A.astype(np.float16)
Bfp16=B.astype(np.float16)
Cfp16=np.matmul(Afp16,Bfp16)
# do matmul manually
Cfp16_manual=sum(Afp16.reshape((M,K,1))*Bfp16.reshape((1,K,N)),axis=1)
# do matmul manually with upcasting to fp32
Cfp16_manual_upcast=sum((Afp16.astype(np.float32).reshape((M,K,1))*Bfp16.astype(np.float32).reshape((1,K,N))).astype(np.float32),axis=1).astype(np.float16)
print("fp16 ",np.linalg.norm(Cfp16-Cfp16_manual))
print("fp16 with upcasting", np.linalg.norm(Cfp16-Cfp16_manual_upcast))
