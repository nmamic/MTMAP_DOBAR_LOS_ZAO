import numpy as np
import matplotlib.pyplot as plt

class NMF:
    def __init__(self,br_iter=100,br_komp=5,epsilon=1e-6):
        self.br_iter=br_iter
        self.br_komp=br_komp
        self.epsilon=epsilon
        self.residuali=[]

    def Bayes(self,podaci,a):
        br_dokumenata, dim_dokumenata = podaci.shape #npr 1000x90
        
        #inicijalizacija
        np.random.seed(200)  
        self.W=np.random.rand(br_dokumenata, self.br_komp) 
        self.H=np.random.rand(self.br_komp, dim_dokumenata) 
        c=(br_dokumenata+dim_dokumenata)/2+a+1
        br=0
        tol=10000
        v=np.mean(podaci)        
        fi=np.var(podaci)
        b=np.pi*(a-1)*v/(2*self.br_komp)
        l=np.zeros(self.br_komp)
        for i in range(self.br_komp):
                l[i] = (0.5 * np.sum(self.W[:, i] ** 2) + 0.5 * np.sum(self.H[i] ** 2) + b) / c


        while tol > self.epsilon and br < self.br_iter:
            br += 1
            R1 = np.tile(l[:, None], (1, dim_dokumenata))
            R2 = np.tile(l, (br_dokumenata, 1))

            self.H = self.H * (self.W.T @ podaci / (self.W.T @ (self.W @ self.H) + fi * self.H / R1+1e-10 ))
            self.W = self.W * (podaci @ self.H.T / ((self.W @ self.H) @ self.H.T + fi * self.W / R2+1e-10 ))

            l_stari = l.copy()
            for i in range(l.size):
                l[i] = (0.5 * np.sum(self.W[:, i] ** 2) + 0.5 * np.sum(self.H[i] ** 2) + b) / c

            tol = np.max(np.abs((l - l_stari) / l_stari))
            greska=np.linalg.norm(podaci - self.W @ self.H,'fro') / np.linalg.norm(podaci,'fro')
            self.residuali.append(greska)

        B = b / c
        za_izbaciti = []
        for i in range(l.size):
            if (l[i] - B) / B < self.epsilon:
                za_izbaciti.append(i)

        self.W = np.delete(self.W, za_izbaciti, axis=1)
        self.H = np.delete(self.H, za_izbaciti, axis=0)

    def plot(self):
        plt.plot(self.residuali)
        plt.show()     
        
    
    def is_nmf(self, podaci, W = None, H = None, suppress_print = False):
        m, n = podaci.shape

        if (W is None or H is None):
            W = np.random.rand(m, self.br_komp) + np.ones((m, self.br_komp))
            H = np.random.rand(self.br_komp, n) + np.ones((self.br_komp, n))

        iter = 0 
        converged = False
        residuals = np.zeros(self.br_iter)

        while (iter < self.br_iter and converged == False):

            #update H
            WH = W @ H + 1e-12

            h_broj = W.T @ np.multiply(WH ** (-2), podaci)
            h_naziv = W.T @ (1.0 / WH) + 1e-12
            H = np.multiply(H, np.divide(h_broj, h_naziv))

            #update W
            WH = W @ H + 1e-12

            w_broj = np.multiply(WH ** (-2), podaci) @ H.T
            w_naziv = (1.0 / WH) @ H.T + 1e-12
            W = np.multiply(W, np.divide(w_broj, w_naziv))
            
            residuals[iter] = np.linalg.norm(podaci - WH)
            if np.abs(residuals[iter-1] - residuals[iter]) < self.epsilon:
                converged = True

            iter += 1

        if suppress_print == True:
            return {'W' : W, 'H' : H, 'residuals' : residuals}
        else:
            print(f"--- IS-NMF Multiplicative Updates summary ---")
            print(f"Iterations: {iter} / {self.br_iter}")
            print(f"Converged:  {'✅ Yes' if converged else '❌ No'}")
            print(f"Final residual (fro norm): {residuals[iter - 1]:.6e}")
            print(f"{'-'*30}")
            
            return {'W' : W, 'H' : H, 'residuals' : residuals}
        
    def ard_is_nmf(self, podaci, tau, a, W0 = None, H0 = None, pruning_threshold = 1e-1, suppress_print = False):
        n, m = podaci.shape

        if (W0 is None or H0 is None):
            NMF_temp = NMF(br_iter = 50, br_komp = self.br_komp)
            result = NMF_temp.is_nmf(podaci, suppress_print=True)
            W = result["W"]
            H = result["H"]
        else:
            NMF_temp = NMF(br_iter = 50, br_komp = self.br_komp)
            result = NMF_temp.is_nmf(podaci, W=W0, H = H0, suppress_print=True)
            W = result["W"]
            H = result["H"]

        tol = np.inf
        tolerances = []

        c = (n + m)/2.0 + a + 1
        e = 1.0/3.0

        b = (np.pi * (a - 1) * np.mean(podaci))/(2*self.br_komp)
        B = b/c

        l = ( 0.5 * ( np.linalg.norm(W, axis = 0)**2 + np.linalg.norm(H, axis = 1)**2 ) + b ) / c
        lambdas = [l]

        iter = 0
        while (tol > tau and iter < self.br_iter):
            #update H
            WH = W @ H + 1e-12

            h_broj = W.T @ ( WH**(-2) * podaci )
            h_naziv = W.T @ (1.0 / WH) + np.divide(H, np.tile(l[:, None], (1, m))) + 1e-12
            H = np.multiply(H, np.power(np.divide(h_broj, h_naziv), e))

            #update W
            WH = W @ H + 1e-12

            w_broj = (WH**(-2) * podaci) @ H.T
            w_naziv = (1.0 / WH) @ H.T + np.divide(W, np.tile(l, (n, 1))) + 1e-12
            W = np.multiply(W, np.power(np.divide(w_broj, w_naziv), e))

            #update lambdas
            l_new = ( 0.5 * ( np.linalg.norm(W, axis = 0)**2 + np.linalg.norm(H, axis = 1)**2 ) + b ) / c
            lambdas.append(l_new)   

            tol = np.max(np.abs((l_new - l) / (l + 1e-12)))
            tolerances.append(tol)
            
            l = l_new
            iter += 1

        criteria = (l - B) / (B + 1e-12)
        mask = criteria > pruning_threshold

        if suppress_print == True:
            return {"W" : W, "H" : H, "tolerances" : tolerances, "iter" : iter, "lambdas" : lambdas, "mask" : mask}

        else:

                # ---- Summary printout ----
            print("\n--- IS-NMF with Automatic Relevance Determination (ARD) Summary ---")
            print(f"Initial rank (k):            {self.br_komp}")
            print(f"Final effective rank:        {H.shape[0] - np.sum(mask)}")
            print(f"Iterations performed:        {iter} / {self.br_iter}")
            print(f"Convergence tolerance τ:     {self.epsilon:.2e}")
            print(f"Final tolerance achieved:    {tol:.2e}")
            print(f"λ statistics:")
            print(f"    Mean λ:                  {np.mean(l):.4e}")
            print(f"    Min λ:                   {np.min(l):.4e}")
            print(f"    Max λ:                   {np.max(l):.4e}")
            print(f"Pruned components:           {np.sum(~mask)}")
            print(f"Active components retained:  {np.sum(mask)}")
            print(f"{'-'*50}\n")

            return {"W" : W, "H" : H, "tolerances" : tolerances, "iter" : iter, "lambdas" : lambdas, "mask" : mask}

        