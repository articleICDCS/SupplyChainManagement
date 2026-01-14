# Guide d'Installation avec Environnement Virtuel

## 🎯 Problème
Vous voulez installer les dépendances de ce projet sans affecter vos autres projets Python.

## ✅ Solution : Environnement Virtuel

### Méthode 1 : Avec PowerShell (Recommandé)

#### Étape 1 : Créer l'environnement virtuel
```powershell
py -3 -m venv venv
```

#### Étape 2 : Activer l'environnement
Vous avez 3 options :

**Option A** : Utiliser cmd.exe temporairement
```powershell
cmd
venv\Scripts\activate
```

**Option B** : Autoriser l'exécution temporairement (si vous êtes administrateur)
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\venv\Scripts\Activate.ps1
```

**Option C** : Utiliser directement le Python de l'environnement virtuel
```powershell
venv\Scripts\python.exe -m pip install -r requirements.txt
```

#### Étape 3 : Installer les dépendances
Une fois l'environnement activé (vous verrez `(venv)` avant votre prompt), installez :
```powershell
pip install -r requirements.txt
```

#### Étape 4 : Exécuter le code
```powershell
python examples\complete_example.py
```

#### Étape 5 : Désactiver l'environnement (quand vous avez fini)
```powershell
deactivate
```

---

### Méthode 2 : Sans Activer l'Environnement (Plus Simple)

Si l'activation pose problème, utilisez directement le Python de l'environnement :

```powershell
# Créer l'environnement (une seule fois)
py -3 -m venv venv

# Installer les dépendances (utilise le pip de l'environnement)
venv\Scripts\python.exe -m pip install -r requirements.txt

# Exécuter le code (utilise le Python de l'environnement)
venv\Scripts\python.exe examples\complete_example.py
```

---

### Méthode 3 : Avec VS Code (Automatique)

1. Ouvrez VS Code dans ce dossier
2. Appuyez sur `Ctrl+Shift+P`
3. Tapez "Python: Select Interpreter"
4. Choisissez l'environnement virtuel `.\venv\Scripts\python.exe`
5. VS Code activera automatiquement l'environnement

---

## 🔍 Vérification

Pour vérifier que vous utilisez bien l'environnement virtuel :

```powershell
# Devrait afficher le chemin vers venv\Scripts\python.exe
venv\Scripts\python.exe -c "import sys; print(sys.executable)"

# Vérifier les packages installés
venv\Scripts\python.exe -m pip list
```

---

## 📝 Commandes Rapides

### Installation (première fois)
```powershell
py -3 -m venv venv
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Utilisation quotidienne
```powershell
# Exécuter les exemples
venv\Scripts\python.exe examples\complete_example.py

# Exécuter un script personnalisé
venv\Scripts\python.exe mon_script.py

# Lancer les tests
venv\Scripts\python.exe -m pytest tests/
```

---

## 🎓 Pourquoi un Environnement Virtuel ?

✅ **Isolation** : Les packages installés n'affectent pas vos autres projets
✅ **Version Python** : Utilise Python 3.12 même si votre système a Python 2.7
✅ **Reproductibilité** : Garantit que le projet utilise les bonnes versions
✅ **Pas de conflit** : Chaque projet a ses propres dépendances
✅ **Facile à supprimer** : Supprimez juste le dossier `venv/`

---

## ❓ FAQ

**Q: L'environnement virtuel prend combien d'espace ?**
A: Environ 50-100 MB

**Q: Puis-je supprimer l'environnement virtuel ?**
A: Oui, supprimez simplement le dossier `venv/` et recréez-le si nécessaire

**Q: Dois-je créer l'environnement à chaque fois ?**
A: Non, une seule fois suffit. Ensuite, activez-le simplement quand vous travaillez sur le projet

**Q: Que faire si j'ai des erreurs d'installation ?**
A: Essayez :
```powershell
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install -r requirements.txt --no-cache-dir
```

---

## 🚀 Installation Complète en Une Ligne

Si vous êtes pressé (méthode sans activation) :

```powershell
py -3 -m venv venv ; venv\Scripts\python.exe -m pip install --upgrade pip ; venv\Scripts\python.exe -m pip install -r requirements.txt
```

Puis pour exécuter :

```powershell
venv\Scripts\python.exe examples\complete_example.py
```

---

## 📌 Ajout au .gitignore

Le dossier `venv/` est déjà dans `.gitignore`, donc il ne sera pas versionné.

---

**Voilà ! Votre environnement est maintenant isolé et n'affectera pas vos autres projets Python.** 🎉
