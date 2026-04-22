{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOpDj1vQau0s6JEuDgEc5tX",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/geekynadir/semester-8/blob/main/Week6.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "CabmceAMbKju",
        "cellView": "form"
      },
      "outputs": [],
      "source": [
        "# @title\n"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Q1\n"
      ],
      "metadata": {
        "id": "Zph4t2j8b_ew"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Q2\n"
      ],
      "metadata": {
        "id": "ztejuY0xcKrw"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.datasets import fetch_california_housing\n",
        "import pandas as pd\n",
        "\n",
        "housing = fetch_california_housing(as_frame=True)\n",
        "housing_df = housing.frame\n",
        "\n",
        "print(\"California Housing Data (first 5 rows):\\n\")\n",
        "print(housing_df.head())\n",
        "\n",
        "print(\"\\nOriginal columns:\\n\")\n",
        "print(housing_df.columns.tolist())\n",
        "\n",
        "selected_columns = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population']\n",
        "selected_features_df = housing_df[selected_columns]\n",
        "\n",
        "print(\"\\nSelected Features DataFrame (first 5 rows):\\n\")\n",
        "print(selected_features_df.head())\n",
        "\n",
        "print(\"\\nSelected columns:\\n\")\n",
        "print(selected_features_df.columns.tolist())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "zl7hTdStcAhx",
        "outputId": "acab5217-84ff-4431-af7a-9f8030fa30c4"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "California Housing Data (first 5 rows):\n",
            "\n",
            "   MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  Latitude  \\\n",
            "0  8.3252      41.0  6.984127   1.023810       322.0  2.555556     37.88   \n",
            "1  8.3014      21.0  6.238137   0.971880      2401.0  2.109842     37.86   \n",
            "2  7.2574      52.0  8.288136   1.073446       496.0  2.802260     37.85   \n",
            "3  5.6431      52.0  5.817352   1.073059       558.0  2.547945     37.85   \n",
            "4  3.8462      52.0  6.281853   1.081081       565.0  2.181467     37.85   \n",
            "\n",
            "   Longitude  MedHouseVal  \n",
            "0    -122.23        4.526  \n",
            "1    -122.22        3.585  \n",
            "2    -122.24        3.521  \n",
            "3    -122.25        3.413  \n",
            "4    -122.25        3.422  \n",
            "\n",
            "Original columns:\n",
            "\n",
            "['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude', 'MedHouseVal']\n",
            "\n",
            "Selected Features DataFrame (first 5 rows):\n",
            "\n",
            "   MedInc  HouseAge  AveRooms  AveBedrms  Population\n",
            "0  8.3252      41.0  6.984127   1.023810       322.0\n",
            "1  8.3014      21.0  6.238137   0.971880      2401.0\n",
            "2  7.2574      52.0  8.288136   1.073446       496.0\n",
            "3  5.6431      52.0  5.817352   1.073059       558.0\n",
            "4  3.8462      52.0  6.281853   1.081081       565.0\n",
            "\n",
            "Selected columns:\n",
            "\n",
            "['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population']\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.datasets import load_iris\n",
        "import pandas as pd\n",
        "from sklearn.preprocessing import MinMaxScaler, StandardScaler\n",
        "\n",
        "iris = load_iris(as_frame=True)\n",
        "iris_df = iris.frame\n",
        "\n",
        "print(\"\\nIris Data (first 5 rows):\\n\")\n",
        "print(iris_df.head())\n",
        "\n",
        "features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']\n",
        "iris_features = iris_df[features]\n",
        "\n",
        "minmax_scaler = MinMaxScaler()\n",
        "iris_normalized = minmax_scaler.fit_transform(iris_features)\n",
        "iris_normalized_df = pd.DataFrame(iris_normalized, columns=features)\n",
        "\n",
        "print(\"\\nNormalized Features (Min-Max Scaling) - first 5 rows:\\n\")\n",
        "print(iris_normalized_df.head())\n",
        "\n",
        "standard_scaler = StandardScaler()\n",
        "iris_standardized = standard_scaler.fit_transform(iris_features)\n",
        "iris_standardized_df = pd.DataFrame(iris_standardized, columns=features)\n",
        "\n",
        "print(\"\\nStandardized Features (Z-score Scaling) - first 5 rows:\\n\")\n",
        "print(iris_standardized_df.head())\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xt3vk3zvcLmo",
        "outputId": "9a167700-504d-427f-d9c6-afb1320947c3"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Iris Data (first 5 rows):\n",
            "\n",
            "   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  \\\n",
            "0                5.1               3.5                1.4               0.2   \n",
            "1                4.9               3.0                1.4               0.2   \n",
            "2                4.7               3.2                1.3               0.2   \n",
            "3                4.6               3.1                1.5               0.2   \n",
            "4                5.0               3.6                1.4               0.2   \n",
            "\n",
            "   target  \n",
            "0       0  \n",
            "1       0  \n",
            "2       0  \n",
            "3       0  \n",
            "4       0  \n",
            "\n",
            "Normalized Features (Min-Max Scaling) - first 5 rows:\n",
            "\n",
            "   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)\n",
            "0           0.222222          0.625000           0.067797          0.041667\n",
            "1           0.166667          0.416667           0.067797          0.041667\n",
            "2           0.111111          0.500000           0.050847          0.041667\n",
            "3           0.083333          0.458333           0.084746          0.041667\n",
            "4           0.194444          0.666667           0.067797          0.041667\n",
            "\n",
            "Standardized Features (Z-score Scaling) - first 5 rows:\n",
            "\n",
            "   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)\n",
            "0          -0.900681          1.019004          -1.340227         -1.315444\n",
            "1          -1.143017         -0.131979          -1.340227         -1.315444\n",
            "2          -1.385353          0.328414          -1.397064         -1.315444\n",
            "3          -1.506521          0.098217          -1.283389         -1.315444\n",
            "4          -1.021849          1.249201          -1.340227         -1.315444\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "rhszpg6tbLKa"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}