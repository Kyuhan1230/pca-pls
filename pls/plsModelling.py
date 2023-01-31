# Import Library
import math
import scipy.stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import chi2
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from functools import reduce


def is_standardized(data):
    """Check that the input data is standardized data.
    If the mean of the data is 0 and the standard deviation is 1, it is judged as standardized data.

    Args:
        data(numpy array): input data

    Returns:
        True if data is already standardized, otherwise False
    """
    mean_sum = np.mean(data, axis=0).sum()
    mean_cond = mean_sum < 1e-6

    if data.ndim > 1:
        std_multiply = reduce(lambda x, y: x * y, np.std(data, axis=0))
        std_cond = abs(std_multiply - 1.0) < 1e-6

        if all([mean_cond, std_cond]):
            return True
        else:
            return False
    else:
        if mean_cond:
            return True
        else:
            return False


class PLSModel:
    """PLS modellings module
    Args:
        n_components (int, float): Number of Principal Components. Defaults to 0.95
                                   When n_components is set between [0, 1], returned that covers at least the percentage
                                   of variance.
        alpha (float): Significance level

    Attributes:
        model
        stored_data
    """

    def __init__(self, n_components=None, alpha=0.05, fault_detect=False):
        """Initialize with user-defined parameters."""
        self.n_components = n_components
        self.alpha = alpha
        self.fault_detect = fault_detect

        # Initialize Attributes
        self.model = None
        self.x_train = None
        self.y_train = None
        self.stored_data = {}
        self.result_fault_detect = {}

    def _determine_n_components(self, x, y):
        """determine number of component
        Args
            x: (Numpy array): X Train
            y: (Numpy array): Y Train
        """
        self.n_features = x.shape[1]
        n_components = self.n_components

        if n_components is None:
            r2_list = []
            for i in range(self.n_features):
                # Create PLS Model
                model = PLSRegression(n_components=i + 1)
                # Fit Model on Data
                model.fit(x, y)
                # Calculate R2 Score
                r2_list.append(model.score(X=x, y=y))

            diff_r2_list = [r2_list[0] - 0]
            for i in range(len(r2_list) - 1):
                diff_r = r2_list[i + 1] - r2_list[i]
                diff_r2_list.append(diff_r)

                if diff_r > 0.01:
                    n_components = self.n_features - 1
                else:
                    n_components = i + 1
                    break
            if n_components == self.n_features or n_components == 0:
                n_components = self.n_features - 1
            return n_components

        if n_components >= x.shape[1]:
            print(f"n_components can not be more then number of features. "
                  f"n_components is set to {int(self.n_features - 1)}")
            n_components = int(self.n_features - 1)

        # Create Model
        model = PLSRegression(n_components=n_components)
        # Fit Model on Data
        model.fit(x, y)
        # Get n_components
        n_components = int(model.n_components)
        if n_components == self.n_features or n_components == 0:
            n_components = self.n_features - 1
        return n_components

    def _preprocess_x(self, x, x_columns=None, x_index=None):
        """Preprocess Data for modellings
        Args:
            x (array-like: Can be of type Numpy or DataFrame): [NxM] array with columns as features and rows as samples.
            x_columns: label of x/feature. Defaults to None
            x_index: index value of data index. Defaults to None
        Returns:
            x: (Numpy array): Convert to numpy array
            x_columns: Same above
            x_index: Same above
        """
        if isinstance(x, pd.DataFrame):
            x_columns = x.columns.values
            x_index = x.index.values
            x = x.values

        if (x_columns is None) or (len(x_columns) == 0) or (len(x_columns) != x.shape[1]):
            x_columns = [f'X{col + 1}' for col in range(x.shape[1])]

        if isinstance(x_columns, list):
            x_columns = np.array(x_columns)

        if (x_index is None) or (len(x_index) == x.shape[0]):
            x_index = np.arange(x.shape[0])

        if isinstance(x_index, list):
            x_index = np.array(x_index)

        return x, x_columns, x_index

    def _normalize_data(self, x, scaler=None):
        """Normalize Data if is not already standardized.
        Args:
            x (Numpy array): [NxM] array with columns as features and rows as samples.
            scaler (object): Standard Scaler for normalize. Defaults to None
        Returns:
            x (numpy array): x scaled
            scaler (object): Standard Scaler for normalize.
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1).T
        if not is_standardized(x):
            if scaler is None:
                scaler = StandardScaler()
                scaler.fit(x)
            x = scaler.transform(x)
            return x, scaler
        else:
            return x, None

    def fit(self, x, y):
        """Fit PLS on data.
        Args:
            x (array-like: Can be of type Numpy or DataFrame): [NxM] array with columns as features and rows as samples.
            y (array-like: Can be of type Numpy or DataFrame): [NxM] array with columns as features and rows as samples.
        Returns
            model (object): PLS model
            coefficient (DataFrame): loadings
            result_fault_detect (): HotellingsT2, SPE
        """
        # Preprocess X
        x, x_columns, x_index = self._preprocess_x(x=x)
        y, y_columns, y_index = self._preprocess_x(x=y)

        # Normalize X: Get X Avg, Std
        _, x_scaler = self._normalize_data(x=x)
        _, y_scaler = self._normalize_data(x=y)

        # Determine Component
        n_components = self._determine_n_components(x=x, y=y)

        # Create Model
        model = PLSRegression(n_components=n_components)

        # Fit Model on Data
        model.fit(x, y)
        print(f"Number of Component is {model.n_components}")
        # Store Train Data(x, y)
        self.x_train, self.y_train = x, y

        # Store model
        self.model = model

        # Get VIP
        vip = self.calculate_vip(x_columns=x_columns)

        # Store Data
        self._store_data(vip, x_columns, y_columns, x_scaler, y_scaler)

        # Get Coefficient
        coefficient = self.calculate_coefficient()

        if self.fault_detect:
            # Compute T2, SPE about train data
            t2, t2_detect = self.get_hotellingt2(data=x, limit=None)
            spe, spe_detect = self.get_spe(data=x, limit=None)
            self.result_fault_detect['t2'] = {"value": t2, "detect_result": t2_detect}
            self.result_fault_detect['spe'] = {"value": spe, "detect_result": spe_detect}

            # store result of t2, spe
            self.result_fault_detect['spe_mean'] = spe.mean()
            self.result_fault_detect['spe_std'] = spe.std()

        return model, coefficient, self.result_fault_detect

    def transform(self, x):
        """Transform Data with fitted model
        Args:
            x (array-like: Can be of type Numpy or DataFrame): [NxM] array to transform
        Returns:

        """
        # Preprocess X
        x, _, _ = self._preprocess_x(x=x)

        # Load Model
        model = self.model

        # transform data to principal components
        score = model.transform(x)

        return score

    def predict(self, x_test):
        """Predict y
        Arg:
            x_test (array-like: Can be of type Numpy or DataFrame): [NxM] array for test
        Return:
            y_predict (Numpy array): Predicted_y
        """
        # Preprocess X
        x_test, _, _ = self._preprocess_x(x=x_test)

        # Predict y
        y_pred = self.model.predict(x_test)
        if y_pred.ndim != self.y_train.shape[1]:
            y_pred.reshape(-1, self.y_train.shape[1])
        return y_pred

    def get_x_predict(self, x):
        loading_p = self.stored_data['loading_p']
        score_t = self.transform(x=x)
        x_hat = np.matmul(score_t, loading_p.T)
        x_hat = self.stored_data['scaler_x'].inverse_transform(x_hat)
        return x_hat

    def _store_data(self, vip, x_columns, y_columns, x_scaler, y_scaler):
        """Store Data about PLS Model
        1. Store input data
        2. Calculate std of score(a.k.a t, pc)
        3. Store statistic data (std of score, mean of x, std of x)
        Arg:
            score (array): [NxK] k is n_components
            loading_p (array): [MxK] k is n_components, M is n_features
            loading_q (array): [MxK] k is n_components, M is n_features
            weight_star (array): Weight Star
            vip (DataFrame) : Variable Importance in Projection
            x_scaler (object): Scaler fitted train data
            x_columns (array): features label
            y_columns (array): targets label
        """
        # Store input data
        # Get Score(principal Components), Loadings and weights
        loading_p = self.model.x_loadings_
        loading_q = self.model.y_loadings_
        score_t = self.model.x_scores_
        weight_star = self.model.x_rotations_

        stored_data = {'score_t': score_t,
                       'loading_p': loading_p,
                       'loading_q': loading_q,
                       'weight_star': weight_star,
                       'vip': vip,
                       'n_components': score_t.shape[1],
                       'n_features': self.n_features,
                       'x_columns': x_columns,
                       'y_columns': y_columns,
                       'scaler': x_scaler,
                       'y_scaler': y_scaler}

        # Calculate std of score(a.k.a t, pc)
        t_std = [score_t[:, i].std() for i in range(stored_data['n_components'])]

        # Store statistic data (std of score, mean of x, std of x)
        x_mean = stored_data["scaler"].mean_
        if not isinstance(x_mean, np.ndarray):
            x_mean = np.array(x_mean)

        x_std = stored_data["scaler"].scale_
        if not isinstance(x_std, np.ndarray):
            x_std = np.array(x_std)

        # Store statistic data (std of score, mean of x, std of x)
        y_mean = stored_data["y_scaler"].mean_
        if not isinstance(x_mean, np.ndarray):
            y_mean = np.array(y_mean)

        y_std = stored_data["y_scaler"].scale_
        if not isinstance(y_std, np.ndarray):
            y_std = np.array(y_std)

        stored_data['t_std'] = t_std
        stored_data['x_mean'] = x_mean
        stored_data['x_std'] = x_std
        stored_data['y_mean'] = y_mean
        stored_data['y_std'] = y_std

        self.stored_data = stored_data

    def calculate_vip(self, x_columns=None) -> pd.DataFrame:
        """calculate vip"""
        if x_columns is None:
            x_columns = self.stored_data['x_columns']

        t = self.model.x_scores_
        w = self.model.x_weights_
        q = self.model.y_loadings_
        p, h = w.shape
        vips = np.zeros((p,))
        s = np.diag(np.dot(np.dot(np.dot(t.T, t), q.T), q)).reshape(h, -1)
        total_s = np.sum(s)

        for i in range(p):
            weight = np.array([(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)])
            vips[i] = np.sqrt(p * (np.dot(s.T, weight)) / total_s)

        vips = list(vips)
        col = x_columns
        tmp = vips.copy()
        tmp.sort(reverse=True)
        re_col = []

        for i in range(len(col)):
            col_idx = vips.index(tmp[i])
            re_col.append(col[col_idx])

        return pd.DataFrame(vips, index=x_columns, columns=['VIP value'])

    def calculate_coefficient(self, scaled=None):
        """calculate pls coefficients in unscaled form"""
        if scaled is not None:
            coef = [float(c) for c in self.model.coef_]
            coef_columns = self.stored_data['x_columns']
        else:
            # Unscaled Coefficient
            avg_list = self.stored_data["x_mean"]   # x 평균
            std_list = self.stored_data["x_std"]    # x 표준편차
            coef_list = self.model.coef_    # scaled Coefficient

            for i in range(len(std_list)):
                if float(std_list[i]) == 0:
                    std_list[i] = 0.000000000000001

            de_coef: list = coef_list.T / std_list
            sum_decoef = np.sum(coef_list.T * avg_list / std_list, axis=1)
            denorm_c = self.stored_data['y_mean'] - sum_decoef

            coef = list(de_coef[0])
            coef.append(denorm_c[0])
            coef_columns = list(self.stored_data['x_columns']) + ['Const']

        coef_ = {coef_columns[i]: coef[i] for i in range(len(coef))}
        coef_ = pd.DataFrame.from_dict([coef_])

        return coef_

    def detect_fault(self, x, alpha=0.05, limits=None, detect_methods=None):
        """ Detect Fault based on Hotelling's T2 and SPE(Squared Prediction Error, aka DmodX)

        Args:
            x (array-like): target data to detect fault
            alpha (float): Significance level
            limits : As the boundary value for fault detection, it should be entered in the order of T2 and SPE.
                     Defaults to None
            detect_methods (list) : Detect to Fault. Default to ['T2', 'SPE'] both.
                                    T2: Hotelling's T2; SPE: Square Prediction Error(a.k.a DModx)

        Returns:

        """
        if detect_methods is None:
            detect_methods = ['T2', 'SPE']

        # Check FD Methods
        if isinstance(detect_methods, str):
            detect_methods = [detect_methods]

        detect_methods = [method.lower() for method in detect_methods]

        for methods in detect_methods:
            if methods not in ['t2', 'spe']:
                raise ValueError("You should use two methods of fault detection: 't2' or 'spe'.")

        # Set Threshold
        if limits is None:
            t2_limit = scipy.stats.chi2.ppf(1 - alpha, self.stored_data['n_components'])
            spe_limit = self.get_spe_limit()
        else:
            if len(limits) != len(detect_methods):
                raise ValueError("You need to set the threshold value for the number of detection methods.")
            else:
                t2_limit, spe_limit = limits

        detect_result = {}
        # Calculate T2
        if 't2' in detect_methods:
            t2, t2_detect = self.get_hotellingt2(data=x, limit=t2_limit)
            detect_result['t2'] = {"value": t2, "detect_result": t2_detect}
        # Calculate SPE
        if 'spe' in detect_methods:
            spe, spe_detect = self.get_spe(data=x, limit=spe_limit)
            detect_result['spe'] = {"value": spe, "detect_result": spe_detect}

        return detect_result

    def get_hotellingt2(self, data, limit):
        """Calculate T2

        Args:
            data (array): target data to detect fault, can be numpy array or DataFrame
            limit : Hotelling's T2 threshold
        Returns:
            t2 (numpy array): Hotelling's T2 value
        """
        t_std = self.stored_data['t_std']
        n_components = self.stored_data['n_components']

        score = self.transform(x=data)

        t2_df = (score / t_std) * (score / t_std)

        t2 = 0
        for i in range(n_components):
            t2 += t2_df[:, i]

        if isinstance(t2, float):
            t2 = np.array(t2)

        if limit is None:
            limit = scipy.stats.chi2.ppf(1 - self.alpha, n_components)

        # Get Bool
        detect_result = np.repeat(False, len(t2))
        t2_bool = t2 > limit
        detect_result[t2_bool] = True

        return t2, detect_result

    def get_spe(self, data, limit):
        """Calculate SPE(Squared Prediction Error)

        Args:
            data(array-like): target data to detect fault, Can be Numpy array or DataFrame
            limit : SPE threshold. Defaults to None
        Returns:
            spe (numpy array): SPE value
        """
        # Normalize Data
        x_scaled, _ = self._normalize_data(data, scaler=self.stored_data['scaler'])

        # Get Loading
        loading_p = self.stored_data['loading_p']

        # Transform Data to Score
        score = self.transform(data)

        # Reconstruct X_test
        x_predict = np.dot(score, loading_p.T)

        # Calculate X error Square
        x_error = x_scaled - x_predict
        sum_x_error = x_error * x_error

        # Calculate SPE
        spe = 0
        for i in range(self.stored_data['n_features']):
            spe += sum_x_error[:, i]

        if isinstance(spe, float):
            spe = np.array(spe)

        if limit is None:
            limit = self.get_spe_limit(mean=spe.mean(), std=spe.std(), alpha=self.alpha)

        # Get Bool
        detect_result = np.repeat(False, len(spe))
        spe_bool = spe > limit
        detect_result[spe_bool] = True

        return spe, detect_result

    def get_spe_limit(self, mean=None, std=None, alpha=0.05):
        if mean is None:
            mean = self.result_fault_detect['spe_mean']
        if std is None:
            std = self.result_fault_detect['spe_std']

        coefficient1 = (std * std) / (2 * mean)
        coefficient2 = (2 * mean * mean) / (std * std)

        if coefficient2 <= 1:
            coefficient2 = 1

        limits = coefficient1 * chi2.ppf(1-alpha, coefficient2)
        return limits

    def get_contribution(self, x, reference=None, calculation_type='Normalized', max_value=None, min_value=None):
        """Calculate the contribution value

        Args:
            x (array-like): target data to compute contribution
            reference: As a reference value to compare with the target, use the statistics of the learning model.
                       Defaults to None
            calculation_type (str): Defaults to 'Normalized'
            max_value (float): Maximum value of contribution value
            min_value (float): Minimum value of contribution value

        Returns:
            contribution (dict): Contribution Value
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        # Get Average in input data
        x_avg = x.mean(axis=0)

        # Normalize data
        x_avg_scaled, _ = self._normalize_data(x=x_avg, scaler=self.stored_data['scaler'])

        # Get Reference
        if reference is None:
            ref_mean = self.stored_data['x_mean']
        else:
            if not isinstance(reference, np.ndarray):
                reference = np.array(reference)
            ref_mean = reference.mean(axis=0)
        ref_scaled, _ = self._normalize_data(x=ref_mean, scaler=self.stored_data['scaler'])

        # Calculate Contribution
        weight_p_w = self.stored_data['weight_star']
        t_std = self.stored_data['t_std']
        columns = self.stored_data['x_columns']

        contribution = {}
        if calculation_type == "Weight 1":
            # ct = (Xj1_normalized - Xj2_normalized) * Weight * Std(t1) * Std(t1)
            # ct = (Xj1_normalized - Xj2_normalized) * Loading * Std(t1) * Std(t1)
            for j, col in enumerate(columns):
                ct = (x_avg_scaled[0][j] - ref_scaled[0][j]) * abs(weight_p_w.T[j, 0]) * t_std[0] * t_std[0]
                if (max_value is not None) and (ct > max_value):
                    ct = max_value
                if (min_value is not None) and (ct < min_value):
                    ct = min_value
                contribution[col] = ct
            return contribution

        elif calculation_type == "Normalized":
            # ct = (Xj1_normalized - Xj2_normalized)
            for j, col in enumerate(columns):
                ct = (x_avg_scaled[0][j] - ref_scaled[0][j])
                contribution[col] = ct
            return contribution
        else:
            raise TypeError("The Calculate Type(arg: calc_type) should be one of ['Weight 1', 'Normalized']")
        pass

    def scatterplot(self, y_true=None, y_predict=None,
                    fig_size=(5, 5), color='blue', trend_line=False,
                    marker_size=None, opacity=0.7, font_size=15, title="PLS Target vs Predict"):
        if y_true is None:
            y_true = self.y_train
        if y_predict is None:
            y_predict = self.model.predict(self.x_train)
        y_true, _, _ = self._preprocess_x(x=y_true)
        y_predict, _, _ = self._preprocess_x(x=y_predict)

        plt.figure(figsize=fig_size)
        plt.scatter(y_true, y_predict, s=marker_size, alpha=opacity, color=color)
        if trend_line:
            z = np.polyfit(y_true.flatten(), y_predict.flatten(), 1)  # (X,Y,차원) 정의
            p = np.poly1d(z)
            plt.plot(y_true, p(y_true), '--', color='gray')
        plt.xlabel('Predicted', size=font_size)
        plt.ylabel('Target', size=font_size)
        plt.title(f"{title}", size=font_size * 1.2)
        plt.show()

    def barplot(self, fig_size=(9, 4), data=None, labels=None,
                color='green', bar_width=0.5, font_size=15, text_rotation=0, title="Contribution Plot"):
        """plot a bar graph of contribution values, Vip or Coefficient

        Args:
            fig_size:
            data:
            labels:
            color:
            bar_width:
            font_size:
            text_rotation:
            title:

        Returns:

        """
        if labels is None:
            labels = self.stored_data['x_columns']

        x_num = len(labels)
        index = np.arange(x_num)
        if data is None:
            print("A random value is used.")
            data = np.random.randn(x_num)
        if isinstance(data, pd.DataFrame):
            data = data.values
        if isinstance(data, {}.values().__class__):
            data = np.array(list(data))
        data = data.flatten()

        plt.figure(figsize=fig_size)
        plt.bar(index + bar_width * 1, data, bar_width, color=color)
        plt.axhline(y=0, color='k')
        plt.xticks(np.arange(bar_width, x_num + bar_width, 1), labels, fontsize=font_size, rotation=text_rotation)
        plt.xlabel('Variable', size=font_size)
        plt.ylabel('Value', size=font_size)
        plt.title(f"{title}", size=font_size*1.2)
        plt.show()

    def biplot(self, plot_score=True, plot_loading=True, x_axis='pc1', y_axis='pc2',
               fig_size=(5, 5), color='blue', text_color='green', oval_color='gray',
               marker_size=None, opacity=1, font_size=15, title="PLS biplot",
               score_test=None, test_color='red'):
        """ Plot PLS biplot.
        The biplot consists of a score plot that expresses the distribution of principal components
        and a loading plot that expresses the relationship between variables.

        Args:
            plot_score:
            plot_loading:
            x_axis:
            y_axis:
            fig_size:
            color:
            text_color:
            oval_color:
            marker_size:
            opacity:
            font_size:
            title:
            score_test:
            test_color:

        Returns:

        """
        if self.stored_data['n_components'] == 1:
            raise ValueError("Score plots cannot be drawn when there is only one principal component.")
        if not isinstance(x_axis, str):
            x_axis = str(x_axis)
        x_axis = int(x_axis.lower().replace('pc', '')) - 1
        if x_axis < 0:
            x_axis = 0

        if not isinstance(y_axis, str):
            y_axis = str(y_axis)
        y_axis = int(y_axis.lower().replace('pc', ''))-1
        if y_axis < 0:
            y_axis = 0

        plt.figure(figsize=fig_size)
        if plot_score:
            score = self.stored_data['score_t']
            xs = score[:, x_axis]
            ys = score[:, y_axis]
            oval_border_x, oval_border_y = self._get_score_oval_border()
            plt.scatter(xs, ys, s=marker_size, color=color, alpha=opacity)
            plt.plot(oval_border_x, oval_border_y, color=oval_color, linewidth=0.5)
            if score_test is not None:
                xs_test = score_test[:, x_axis]
                ys_test = score_test[:, y_axis]
                plt.scatter(xs_test, ys_test, s=marker_size, color=test_color, alpha=1)

        if plot_loading:
            loading = self.stored_data['loading_p'].T
            labels = self.stored_data['x_columns']
            loading_y = self.stored_data['loading_q'].T
            labels_y = self.stored_data['y_columns']

            for i in range(len(labels)):
                plt.arrow(0, 0, loading[x_axis, i], loading[y_axis, i], color='pink', alpha=0.5)
                plt.text(loading[x_axis, i] * 1.15, loading[y_axis, i] * 1.15, labels[i],
                         color=text_color, ha='center', va='center')
            plt.arrow(0, 0, loading_y[x_axis, 0], loading_y[y_axis, 0], color='r', alpha=1)
            plt.text(loading_y[x_axis, 0] * 1.15, loading_y[y_axis, 0] * 1.15, labels_y[0],
                     color='red', ha='center', va='center')

            if not plot_score:
                plt.xlim(-1, 1)
                plt.ylim(-1, 1)

        plt.xlabel(f"PC{x_axis+1}", fontsize=font_size)
        plt.ylabel(f"PC{y_axis+1}", fontsize=font_size)
        plt.grid()
        plt.title(f"{title}", size=font_size * 1.2)
        plt.show()

    def _get_score_oval_border(self):
        """Computes the elliptical bounds of the SCore Plot.
        Calculated only for PC1, PC2.

        Returns:
            oval_border_x
            oval_border_y
        """
        data_size = self.stored_data['score_t'].shape[0]
        t1_std, t2_std = self.stored_data['t_std'][0], self.stored_data['t_std'][1]

        dfn = 2
        dfd = data_size - 2

        f = scipy.stats.f.ppf(0.95, dfn, dfd)
        ff = 2 * ((data_size - 1) / (data_size - 2))

        a = math.sqrt(t1_std * t1_std * f * ff)
        b = math.sqrt(t2_std * t2_std * f * ff)
        angle = np.linspace(0, 2 * np.pi, 100)

        oval_border_x = a * np.cos(angle)
        oval_border_y = b * np.sin(angle)

        return [oval_border_x, oval_border_y]

    def plot_anomaly_score(self, anomaly_score=None, detect_method=None, limit=None,
                           concat=False,
                           fig_size=(9, 3), color='blue', oval_color='red',
                           marker_size=None, opacity=1, line_width=1,
                           font_size=15):
        """Plot the time series trend of Anomaly Score (T2, SPE) values.

        Args:
            anomaly_score:
            detect_method:
            limit:
            concat:
            fig_size:
            color:
            oval_color:
            marker_size:
            opacity:
            line_width:
            font_size:

        """
        detect_method = detect_method.lower()
        if isinstance(anomaly_score, pd.DataFrame):
            anomaly_score = anomaly_score.values

        if 't2' == detect_method:
            if limit is None:
                t2_limit = scipy.stats.chi2.ppf(1 - self.alpha, self.stored_data['n_components'])
            else:
                t2_limit = limit
            if anomaly_score is None:
                value = self.result_fault_detect['t2']['value']
                len_train = len(value)
            else:
                if concat:
                    train_value = self.result_fault_detect['t2']['value']
                    len_train = len(train_value)
                    test_value = anomaly_score
                    value = np.concatenate((train_value, test_value), axis=0)
                else:
                    len_train = 0
                    value = anomaly_score

            fig, ax = plt.subplots(figsize=fig_size)
            ax.plot(value, color=color, linewidth=line_width)
            ax.scatter(x=np.arange(len(value)), y=value, s=marker_size, color=color, alpha=opacity)
            ax.set_xlim(0, len(value))
            ax.set_ylabel('T2', fontsize=font_size)
            ax.set_title(f"Hotelling's T2 Trend", fontsize=font_size)
            ax.axhline(y=t2_limit, color=oval_color)
            ax.axvline(x=len_train, color='gray')
            plt.tight_layout()
            plt.show()

        elif 'spe' in detect_method:
            if limit is None:
                spe_limit = self.get_spe_limit()
            else:
                spe_limit = limit

            if anomaly_score is None:
                value = self.result_fault_detect['spe']['value']
                len_train = len(value)
            else:
                if concat:
                    train_value = self.result_fault_detect['spe']['value']
                    len_train = len(train_value)
                    test_value = anomaly_score
                    value = np.concatenate((train_value, test_value), axis=0)
                else:
                    len_train = 0
                    value = anomaly_score
            fig, ax = plt.subplots(figsize=fig_size)
            ax.plot(value, color=color, linewidth=line_width)
            ax.scatter(x=np.arange(len(value)), y=value, s=marker_size, color=color, alpha=opacity)
            ax.set_xlim(0, len(value))
            ax.set_ylabel('SPE', fontsize=font_size)
            ax.set_title(f"SPE Trend", fontsize=font_size)
            ax.axhline(y=spe_limit, color=oval_color)
            ax.axvline(x=len_train, color='gray')
            plt.tight_layout()
            plt.show()

        else:
            raise ValueError("You should use two methods of fault detection: 't2' or 'spe'.")
