# if MX - B > 0 then 1 else 0
# 0 <= B <= m2
# Dans notre implémentation on va faire différement pour calculer les gradients plus facilement

# on veut faire en sorte qu'à chaque fois que l'on fait un produit de matrice alors on quantise les outputs
# on soustrait des biais quantisé de la bonne dimension

# M de dim (m1, m2)
# X de dim (m2, m3)
# b de dim (m1)

# On fait MX/m2 - b > 0 then 1 else 0
# 0 <= b <= 1 (on initialise les poids vers 1/2 pour b)
# Pour la backprop on fait tout comme s'il n'y avait que l'opération MX/m2 - b (sans le seuil) pour M et b
# Pour l'input on essaie les 2 versions, une avec le seuil pris en compte et l'autre sans le seuil pris en compte

# On supprime tous les layers de normalisation
# On supprime tous les les layers de softmax ?
