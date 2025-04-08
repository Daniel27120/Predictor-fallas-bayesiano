import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import random
from deap import base, creator, tools
import os

# Ruta del archivo CSV
ruta = "fallas_componentes.csv"

# Función para entrenar el modelo
def entrenar_modelo(alpha):
    data = pd.read_csv(ruta)
    
    # Eliminar la columna ID si existe
    if 'ID' in data.columns:
        data = data.drop(columns=['ID'])
    
    # Usar las 2 primeras columnas como síntomas y la tercera como variable objetivo
    # (La cuarta columna 'Probabilidad' se ignora en el entrenamiento)
    symptom_columns = data.columns[:-2]          # ['Sintoma_1', 'Sintoma_2']
    target_column = data.columns[-2]               # 'Falla_Probable'
    
    # Codificadores para síntomas y para la falla
    le_symptoms = {col: LabelEncoder().fit(data[col]) for col in symptom_columns}
    for col in symptom_columns:
        data[col] = le_symptoms[col].transform(data[col])
    
    le_target = LabelEncoder()
    data[target_column] = le_target.fit_transform(data[target_column])
    
    # Entrenar el modelo
    X = data[symptom_columns]
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Devolver modelo y elementos necesarios
    return model, le_symptoms, le_target, symptom_columns, target_column, accuracy

# Función para volver a entrenar el modelo con los datos actuales
def volver_a_entrenar(model, le_symptoms, le_target, symptom_columns):
    print("\n🔄 Entrenando el modelo con los datos actuales...")
    # Usamos alpha=1.0 por defecto
    model, le_symptoms, le_target, symptom_columns, target_column, accuracy = entrenar_modelo(1.0)
    print("✅ Modelo entrenado nuevamente.")
    return model, le_symptoms, le_target, symptom_columns, target_column, accuracy

# Función para optimización genética (se mantiene igual)
def optimizar():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximizar la precisión
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    # Rango para alpha entre 0.001 y 10
    toolbox.register("attr_float", random.uniform, 0.001, 10)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    population = toolbox.population(n=10)
    
    def evaluar(individuo):
        alpha = max(0.001, individuo[0])
        model, le_symptoms, le_target, symptom_columns, target_column, accuracy = entrenar_modelo(alpha)
        return accuracy,
    
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=1, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluar)
    
    ngen = 200
    cxpb = 0.5
    mutpb = 0.2
    
    for gen in range(ngen):
        print(f"\nGeneración {gen+1}/{ngen}")
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
    
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
                
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values
                
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)
    
        population[:] = offspring
    
    mejor_individuo = tools.selBest(population, 1)[0]
    mejor_alpha = mejor_individuo[0]
    print(f"\nMejor valor de alpha encontrado: {mejor_alpha}")
    
    model, le_symptoms, le_target, symptom_columns, target_column, accuracy = entrenar_modelo(mejor_alpha)
    print(f"Precisión con el mejor alpha encontrado: {mejor_alpha}")

# Función principal
def main():
    model, le_symptoms, le_target, symptom_columns, target_column, accuracy_actual = entrenar_modelo(1.0)
    print(f"Precisión inicial del modelo: {accuracy_actual}")
    
    while True:
        print("\n¿Qué deseas hacer?")
        print("1. Predecir una falla")
        print("2. Agregar un nuevo registro de síntomas y falla")
        print("3. Volver a entrenar el modelo con los datos actuales")
        print("4. Optimización genética de los parámetros")
        print("5. Salir del programa")
        opcion = input("Selecciona una opción (1, 2, 3, 4 o 5): ")
    
        if opcion == "1":
            # Predecir una falla a partir de 2 síntomas
            print("\nIntroduce 2 síntomas para predecir la falla:")
            # Solo mostramos y pedimos los 2 primeros síntomas
            for col in symptom_columns:
                print(f" - {col}: {list(le_symptoms[col].classes_)}")
    
            user_input = []
            for col in symptom_columns[:2]:
                valor = input(f"Ingrese el valor para '{col}': ")
                while valor not in le_symptoms[col].classes_:
                    print(f"Opción inválida. Valores permitidos: {list(le_symptoms[col].classes_)}")
                    valor = input(f"Ingrese el valor para '{col}': ")
                codificado = le_symptoms[col].transform([valor])[0]
                user_input.append(codificado)
    
            # Rellenar con 0 para que la entrada tenga el mismo número de características que se entrenó el modelo
            while len(user_input) < len(symptom_columns):
                user_input.append(0)
    
            # Obtener las probabilidades para cada clase
            proba = model.predict_proba([user_input])[0]
            # Se selecciona la clase con mayor probabilidad:
            idx_max = proba.argmax()
            falla_predicha = le_target.inverse_transform([model.classes_[idx_max]])[0]
            probabilidad_predicha = proba[idx_max]
    
            print(f"\n⚠️  Falla probable predicha: {falla_predicha}")
            print(f"Probabilidad: {probabilidad_predicha:.2f}")
    
        elif opcion == "2":
            print("\nIntroduce un nuevo registro (síntomas + falla final):")
            nuevo_registro = {}
            for col in symptom_columns:
                valor = input(f"Ingrese el valor para '{col}': ")
                nuevo_registro[col] = valor
    
            falla = input(f"Ingrese la falla correspondiente ({target_column}): ")
            nuevo_registro[target_column] = falla
    
            df_original = pd.read_csv(ruta)
            df_original = df_original.drop(columns=['ID'], errors='ignore')
    
            df_original.loc[len(df_original)] = nuevo_registro
            df_original.to_csv(ruta, index=False)
    
            print("\n✅ Nuevo registro agregado a la base de datos.")
            print("Puedes volver a correr el programa para reentrenar el modelo con este nuevo dato.")
    
        elif opcion == "3":
            model, le_symptoms, le_target, symptom_columns, target_column, accuracy_actual = volver_a_entrenar(model, le_symptoms, le_target, symptom_columns)
    
        elif opcion == "4":
            optimizar()
    
        elif opcion == "5":
            print("\n🔴 Saliendo del programa. ¡Hasta luego!")
            break
    
        else:
            print("❌ Opción no válida.")
    
if __name__ == "__main__":
    main()
