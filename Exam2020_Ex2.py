# Exercice 2

# Ci-dessous l'integralite des packages necessaires (d'apres ma correction)
import numpy as np
from scipy.optimize import minimize

# Creer la fonction _portfolio_risk(weights, covariances)
def _portfolio_risk(weights, covariances):
    
    # Calculer portfolio_risk, le risque (la volatilite) du portefeuille d'apres l'Equation (7) en utilisant les weights et la matrices de covariances
    portfolio_risk = np.sqrt(np.dot(np.dot(weights, covariances), weights.T))

    # La fonction _portfolio_risk(weights, covariances) renvoie la volatilite du portefeuille (nommee portfolio_risk)
    return portfolio_risk

# Creer la fonction _assets_risk_contribution_to_portfolio_risk(weights, covariances)
def _assets_risk_contribution_to_portfolio_risk(weights, covariances):

    # Calculer portfolio_risk, le risque du portefeuille en utilisant la fonction _portfolio_risk(weights, covariances)
    portfolio_risk = _portfolio_risk (weights, covariances)
    
    # Calculer assets_risk_contribution de dimension (nb_assets,1), la contribution de chaque actif au risque du portefeuille en utilisant l'Equation (8) 
    assets_risk_contribution = np.multiply(weights.T, covariances * weights.T) / portfolio_risk

    # La fonction _assets_risk_contribution_to_portfolio_risk(weights, covariances) renvoie la contribution de chaque actif au risque du portefeuille (nommee assets_risk_contribution)
    return assets_risk_contribution

# Creer la fonction _risk_budget_objective_error(weights, args)
def _risk_budget_objective_error(weights, args):
    
    # Recuperer la matrice de covariance qui occupe la premiere position dans la variable args
    covariances = args[0]
    
    # Recuperer la contribution desiree de chaque actif au risque du portefeuille, assets_risk_budget occupe la seconde position dans la variable args 
    assets_risk_budget = args[1]
    
    # Convertir la variable weights de type array en une matrice nommee egalement weights (indice: utiliser la fonction np.matrix())
    weights = np.matrix(weights)
    
    # Calculer portfolio_risk, le risque du portefeuille en utilisant la fonction _portfolio_risk(weights, covariances)
    portfolio_risk = _portfolio_risk(weights, covariances)
    
    # Calculer assets_risk_contribution, la contribution de chaque actif au risque du portefeuille en utilisant la fonction _assets_risk_contribution_to_portfolio_risk(weights, covariances)
    assets_risk_contribution = _assets_risk_contribution_to_portfolio_risk(weights, covariances)
    
    # Calculer assets_risk_target (une matrice de dimension (nb_assets,1)), la contribution désiree de chaque actif au risque du portefeuille en multipliant portfolio_risk et asset_risk_budget
    assets_risk_target = np.multiply(portfolio_risk, assets_risk_budget)
    
    # Calculer l'erreur, nommee error) entre la contribution desiree (assets_risk_target) et la contribution realisee (assets_risk_contribution) de chaque actif
    error = sum(np.square(assets_risk_contribution - assets_risk_target.T))
    
    # La fonction _risk_budget_objective_error(weights, args) renvoie l'erreur calculee (nommee error)
    return error

# Creer la fonction _get_risk_parity_weights(covariances, assets_risk_budget, initial_weights)
def _get_risk_parity_weights(covariances, assets_risk_budget, initial_weights):
    
    # Creer constraints contenant 2 contraintes :
    # 1. Une egalite puisque la somme des poids est egale a 1
    # 2. Une inegalite puisque toute les positions sont longues, les poids sont donc superieur a 0
    contraintes = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                   {'type': 'ineq', 'fun': lambda x: x})

    # Creer optimize_result contenant les resultat de l'optimisation realisee avec la fonction minimize du package scipy.optimize
    # 1er argument : la fonction a optimiser
    # 2eme argument : les poids initiaux
    # 3eme argument : args contenant la matrice de covariance (covariances) et la contribution desiree de chaque actif au risque du portefeuille (assets_risk_budget)
    # 4eme argument : la methode d'optimisation
    # 5eme argument : les contraintes de constraints
    # 6eme argument : le parametre de tolerance pour l'arret de l'optimisation
    optimize_result = minimize(_risk_budget_objective_error, 
                               initial_weights, 
                               [covariances, assets_risk_budget], 
                               method='SLSQP',  
                               constraints = contraintes,
                               tol = tolerance)

    # Recoperer dans weights les poids obtenus a la suite de l'optimisation (aller les chercher dans optimize_result)
    weights = optimize_result.x
    
    # La fonction _get_risk_parity_weights(covariances, assets_risk_budget, initial_weights) renvoie les poids optimaux (nommee weights)
    return weights

# DEBUTER ICI VOTRE PROGRAMMATION
# Definir le 6eme parametre, nomme tolerance, le parametre de tolerance pour l'arret de l'optimisation est fixe a 1e-20
tolerance = 1e-20

# Definir le nombre d'actifs, nomme nb_assets, dans cet exemple ce nombre est egal a 4
nb_assets = 4

# Definir la matrice de covariance, nommmee covariances, un array de dimension (nb_assets,nb_assets) d'apres l'enonce de l'Exercice 3
covariances = np.array([[.01, .02, .03, .04],
                        [.02, .04, .06, .08],
                        [.03, .06, .09, .12],
                        [.04, .08, .12, .16]])

# Definir la contribution desiree de chaque actif au risque du portefeuille, nommee assets_risk_budget, un array de dimension (nb_assets,)
# Ici nous souhaitons que la contribution de chaque actif soit egal (la somme de ces contributions est egale a 1)
assets_risk_budget = np.ones(nb_assets) / nb_assets

# Initialisation de la matrice des poids, nommee init_weights, un array de dimension (nb_assets,)
# Ici nous souhaitons debuter l'optimisation avec des poids equipondere
init_weights = np.ones(nb_assets) / nb_assets

# Recuperer dans RiskParity_weights le resultat du processus d'optimisation sur les poids en appelant la fonction _get_risk_parity_weights(covariances, assets_risk_budget, init_weights)
RiskParity_weights = _get_risk_parity_weights(covariances, assets_risk_budget, init_weights)
