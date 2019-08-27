Federated Learning and several improved versions, implemented using:
- Python 3.7.3
and packages including:
- Pytorch 1.1.0
- PySyft 0.1.2
- numpy 1.16.4

**SAFA Protocol**
- Local update:  
              w_i(t) = clientUpdate(i, w_i(t-1)), if i in M-K(t)

- Pre-aggregation Cache update:  
              w*_k(t) = w_k(t), if k in Union_(v=t-tao->t){Pv}    // picked, relatively up-to-date models 

- SAFA aggregation:  
              W(t) = sum_(k in M){n_k/n * w*_k(t)}     
           
- Post-aggregation Cache update:  
              w*_k(t+1) = w_k(t), if k in Union_(v=t-tao->t){Qv}  // undrafted, relatively up-to-date models  
              w*_k(t+1) = W(t), if k in Union_(v<t-tao){PvUQv}    // picked/undrafted, deprecated models   
              w*_k(t+1) = w*_k(t), if k in K                      // no update for crashed clients  
- Model distribution:  
              w_i(t) = W(t), if i in M-K                          // everybody who submits will receive and sync.  
              w_j(t) = w_j(t-1), if j in K                        // crashed clients stay async.  

where Pv/Qv is a set of picked/undrafted, version-v models/clients, P = U(Pv) is picked set, Q = U(Qv) is undrafted set, M is the client set, K is the crash set, tao denotes lag tolerance.
