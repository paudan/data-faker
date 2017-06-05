import sys
import pandas as pd
import numpy as np
from random import getrandbits
from collections import OrderedDict
from argparse import ArgumentParser
from datetime import datetime
import ruamel.yaml as yaml
from faker import Factory
from faker.providers.date_time import Provider as date_provider
from faker.providers.currency import Provider as currency_provider
from dateutil import parser


class ConfigurationException(Exception):

    def __init__(self, message, conf_file):
        super(ConfigurationException, self).__init__("Error while parsing configuration %s: %s" %
                                                     (conf_file if conf_file is not None else 'None', message))
        self.conf_file = conf_file


def _get_param(_params, name, default=None):
    if name is None:
        return None
    return _params[name] if _params is not None and _params.has_key(name) else default


def _get_dist_param(column, default=None):
    _params = _get_param(column, 'params')
    if not _params:
        return None
    _params = _get_param(_params, 'distribution')
    if not _params and default is not None:
        return {'type': default}
    return _params


def _generate_range(column, min_, max_, dtype, length, conf_file=None):
    if min_ is None and max_ is None:
        return
    _params = _get_param(column, 'params')
    _from = _get_param(_params, 'from', min_)
    _to = _get_param(_params, 'to', max_)
    if (min_ is not None and (_from < min_ or _to < min_)) or (max_ is not None and (_from > max_ or _to > max_)):
        raise ConfigurationException('Range must be between %d and %d' % (min_, max_), conf_file)
    if _from > _to:
        raise ConfigurationException("Invalid range: 'from' value must be less or equal to 'to' value", conf_file)
    return np.random.randint(low=_from, high=_to, dtype=dtype, size=length)


def _generate_gaussian(column, dtype, length):
    if dtype is not None and np.issubdtype(dtype, np.number):
        _params = _get_param(column, 'params')
        if _params:
            _min = _get_param(_params, 'min')
            _max = _get_param(_params, 'max')
        if np.issubdtype(dtype, np.integer):
            _min = _min if _min is not None else np.iinfo(dtype).min
            _max = _max if _max is not None else np.iinfo(dtype).max
        elif np.issubdtype(dtype, np.float):
            _min = _min if _min is not None else np.finfo(dtype).min
            _max = _max if _max is not None else np.finfo(dtype).max
        else:
            _min = _min if _min is not None else 0
            _max = _max if _max is not None else 0
        _dist = _get_dist_param(column, 'Gaussian')
        _dist = _get_param(_dist, 'type') if _dist else None
        if _dist and _dist.lower() == 'gaussian':
            if issubclass(dtype, np.integer):
                return np.random.randint(low=_min, high=_max, dtype=dtype, size=length)
            else:
                values = _min + np.random.rand(length, 1) * (_max - _min)
                return values.flatten()
        return None


def _generate_text(column, length, func):
    _params = _get_param(column, 'params')
    if _params:
        _locale = _get_param(_params, 'locale')
        fake = Factory.create(_locale) if _locale else Factory.create()
        if _params:
            _list = _get_param(_params, 'list')
            if not _list:
                _count = _get_param(_params, 'count')
                if _count is not None:
                    _list = [getattr(fake, func)() for i in range(_count)]
                else:
                    _list = [getattr(fake, func)().encode(encoding='UTF-8') for i in range(length)]
    else:
        fake = Factory.create()
        _list = [getattr(fake, func)().encode(encoding='UTF-8') for i in range(length)]
    return [_list[i] for i in np.random.randint(low=0, high=len(_list), size=length)]


def _generate_distribution(column, length, dtype=None):
    _params = _get_dist_param(column)
    if _params:
        _dist = _get_param(_params, 'type')
        if _dist is None:
            return
        _params = _get_param(_params, 'params')
        if _dist.lower() == 'gaussian':
            _mean = _get_param(_params, 'mean')
            _sigma = _get_param(_params, 'sigma')
            if _mean is None and _sigma is None:
                return _generate_gaussian(column, dtype, length)
            else:
                return np.random.normal(loc=_mean if _mean is not None else 0,
                                        scale=_sigma if _sigma is not None else 1.0, size=length)
        elif _dist.lower() == 'lognormal':
            _mean = _get_param(_params, 'mean')
            _sigma = _get_param(_params, 'sigma')
            return np.random.lognormal(mean=_mean if _mean is not None else 0,
                                       sigma=_sigma if _sigma is not None else 1.0, size=length)
        elif _dist.lower() == 'poisson':
            _lambda = _get_param(_params, 'lambda')
            return np.random.poisson(lam=_lambda, size=length)
        elif _dist.lower() == 'beta':
            _a = _get_param(_params, 'a')
            _b = _get_param(_params, 'b')
            return np.random.beta(a=_a, b=_b, size=length)
        elif _dist.lower() == 'binomial':
            _n = _get_param(_params, 'n')
            _p = _get_param(_params, 'p')
            return np.random.binomial(n=_n, p=_p, size=length)
        elif _dist.lower() == 'gamma':
            _gamma = _get_param(_params, 'gamma')
            _scale = _get_param(_params, 'scale')
            return np.random.gamma(shape=_gamma, scale=_scale if _scale is not None else 0, size=length)
        elif _dist.lower() == 'uniform':
            _low = _get_param(_params, 'low')
            _high = _get_param(_params, 'high')
            return np.random.uniform(low=_low, high=_high, size=length)
        elif _dist.lower() == 'chi-square':
            _df = _get_param(_params, 'df')
            return np.random.chisquare(df=_df, size=length)
        elif _dist.lower() == 'weibull':
            _a = _get_param(_params, 'a')
            return np.random.weibull(a=_a, size=length)
        elif _dist.lower() == 'triangular':
            _left = _get_param(_params, 'left')
            _mode = _get_param(_params, 'mode')
            _right = _get_param(_params, 'right')
            return np.random.triangular(left=_left, mode=_mode, right=_right, size=length)


def _validate_configuration(conf_file):

    def check_range(a, value, range, _label, conf_file):
        if range is not None:
            a_min, a_max = range
            if value < a_min or value > a_max:
                raise ConfigurationException("Feature '{3}': '{0}' parameter must be in range [{1}; {2}]"
                                             .format(a, a_min, a_max, _label), conf_file)

    def check_params(_params, a, b, _str, a_optional=False, b_optional=False,
                              a_range=None, b_range=None, check_greater=False, check_positive=True):
        if _params is None:
            return
        _params = _get_param(_params, 'params')
        if _params is None:
            return
        _a = _get_param(_params, a)
        _b = _get_param(_params, b)
        if not a_optional and _a is None:
            raise ConfigurationException("Feature '{2}': '{0}' parameter must be set for {1}"
                                         .format(a, _str, _label), conf_file)
        if not b_optional and _b is None:
            raise ConfigurationException("Feature '{2}': '{0}' parameter must be set for {1}"
                                         .format(b, _str, _label), conf_file)
        if check_positive and ((_a is not None and _a < 0) or (_b is not None and _b < 0)):
            raise ConfigurationException("Feature '{3}': '{0}' and '{1}' parameters for {2} must be positive"
                                         .format(a, b, _str, _label), conf_file)
        check_range(a, _a, a_range, _label, conf_file)
        check_range(b, _b, b_range, _label, conf_file)
        if check_greater and _a > _b:
            raise ConfigurationException("Feature '{2}': value of '{0}' parameter must be less than value of '{1}'"
                                         .format(a, b, _label), conf_file)

    def check_single_param(_params, a, _str, a_optional=False, check_positive=True):
        if _params is None:
            return
        _params = _get_param(_params, 'params')
        if _params is None:
            return
        _a = _get_param(_params, a)
        if not a_optional and _a is None:
            raise ConfigurationException("Feature '{2}': '{0}' parameter must be set for {1}"
                                         .format(a, _str, _label), conf_file)
        if check_positive and (_a is not None and _a < 0):
            raise ConfigurationException("Feature '{3}': '{0}' parameter for {1} must be positive"
                                         .format(a, _str, _label), conf_file)

    with open(conf_file, 'r') as stream:
        try:
            conf = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ConfigurationException(exc.message, conf_file)
    columns = _get_param(conf, 'columns')
    if not columns:
        raise ConfigurationException('No columns are defined in the configuration file', conf_file)
    for column in columns:
        _label = _get_param(column, 'name')
        _type = _get_param(column, 'type')
        _dist_params = _get_dist_param(column)
        # If distribution is set then no need to define type (float as default)
        if not _dist_params:
            try:
                _type_num = np.issubdtype(np.dtype(_type).type, np.number)
            except TypeError, ex:
                _type_num = False
            if not _type:
                raise ConfigurationException('Type is not set for {0}:'.format(_label), conf_file)
            if _type not in ['day', 'month', 'weekday', 'year', 'date', 'time', 'name', 'country', 'city', 'company', 'currency', 'boolean'] \
                    and not _type_num:
                raise ConfigurationException('Invalid type for %s:' % _label, conf_file)
        if _type == 'date':
            _params = _get_param(column, 'params')
            _to = parser.parse(_params['to']) if _params and _params.has_key('to') else datetime.now()
            _from = parser.parse(_params['from']) if _params and _params.has_key('from') else _to
            if _from > _to:
                raise ConfigurationException('Invalid date range for {0}: [{1}; {2}]'.format(_from, _to, _label), conf_file)
        else:
            _params = _get_param(column, 'params')
            if not _params:
                continue
            if _params.has_key('min') and _params.has_key('max'):
                _from = _get_param(_params, 'min')
                _to = _get_param(_params, 'max')
                if _from > _to:
                    raise ConfigurationException('Invalid numeric range for {0}: [{1}; {2}]'.format(_from, _to, _label), conf_file)
            if _params.has_key('distribution'):
                _dist = _get_param(_dist_params, 'type')
                if _dist is None:
                    raise ConfigurationException('Distribution type is not set for {0}:'.format(_label), conf_file)
                if _dist.lower() == 'beta':
                    check_params(_dist_params, 'a', 'b', 'beta distribution')
                elif _dist.lower() == 'binomial':
                    check_params(_dist_params, 'n', 'p', 'binomial distribution', b_range=(0, 1))
                elif _dist.lower() == 'gamma':
                    check_params(_dist_params, 'gamma', 'scale', 'Gamma distribution')
                elif _dist.lower() == 'uniform':
                    check_params(_dist_params, 'low', 'high', 'uniform distribution', check_greater=True)
                elif _dist.lower() == 'chi-square':
                    check_single_param(_dist_params, 'df', 'chi-square distribution')
                elif _dist.lower() == 'poisson':
                    check_single_param(_dist_params, 'lambda', 'Poisson distribution')
                elif _dist.lower() == 'weibull':
                    check_single_param(_dist_params, 'a', 'Weibull distribution')
                elif _dist.lower() == 'lognormal':
                    check_params(_dist_params, 'mean', 'sigma', 'lognormal distribution')
                elif _dist.lower() == 'triangular':
                    check_params(_dist_params, 'left', 'mode', 'triangular distribution', check_greater=True, check_positive=False)
                    check_params(_dist_params, 'mode', 'right', 'triangular distribution', check_greater=True, check_positive=False)


def generate_pandas(conf_file):
    _validate_configuration(conf_file)
    with open(conf_file, 'r') as stream:
        try:
            conf = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ConfigurationException(exc.message, conf_file)

    length = _get_param(conf, 'length', 0)
    columns = _get_param(conf, 'columns')
    data = OrderedDict()
    for column in columns:
        _label = _get_param(column, 'name')
        _type = _get_param(column, 'type')

        if not _label or (not _type and not _get_dist_param(column)):
            continue
        if _type == 'day':
            data[_label] = _generate_range(column, 1, 31, np.uint8, length, conf_file)
        elif _type == 'month':
            data[_label] = _generate_range(column, 1, 12, np.uint8, length, conf_file)
        elif _type == 'weekday':
            data[_label] = [date_provider.day_of_week() for i in range(length)]
        elif _type == 'year':
            data[_label] = _generate_range(column, datetime.min.year, datetime.now().year, np.uint16, length, conf_file)
        elif _type == 'date':
            _params = _get_param(column, 'params')
            _to = parser.parse(_params['to']) if _params and _params.has_key('to') else datetime.now()
            _from = parser.parse(_params['from']) if _params and _params.has_key('from') else _to
            _pattern = _get_param(_params, 'pattern', '%Y-%m-%d')
            data[_label] = [date_provider.date_time_between_dates(_from, _to).strftime(_pattern)
                            for i in range(length)]
        elif _type == 'time':
            _params = column['params']
            _pattern = _get_param(_params, 'pattern', '%H:%M:%S')
            data[_label] = [date_provider.time(pattern=_pattern) for i in range(length)]
        elif _type == 'currency':
            _params = _get_param(column, 'params')
            if _params:
                _list = _get_param(_params, 'list')
                if not _list:
                    _count = _get_param(_params, 'count')
                    if _count:
                        _list = [currency_provider.currency_code() for i in range(_count)]
                _items = [_list[i] for i in np.random.randint(low=0, high=len(_list), size=length)]
            else:
                _items = [currency_provider.currency_code() for i in range(length)]
            data[_label] = _items
        elif _type == 'name':
            data[_label] = _generate_text(column, length, 'name')
        elif _type == 'country':
            _params = _get_param(column, '_params')
            if _params and _get_param(_params, 'code') is not None:
                if _get_param(_params, 'code') == True:
                    data[_label] = _generate_text(column, length, 'country_code')
                else:
                    data[_label] = _generate_text(column, length, 'country')
            else:
                data[_label] = _generate_text(column, length, 'country')
        elif _type == 'city':
            data[_label] = _generate_text(column, length, 'city')
        elif _type == 'company':
            data[_label] = _generate_text(column, length, 'company')
        elif _type == 'boolean':
            _items = [getrandbits(1) for i in range(length)]
            _params = _get_param(column, 'params')
            _as_int = _get_param(_params, 'as_int')
            if _as_int is not None and _as_int:
                pass
            else:
                _items = [bool(item) for item in _items]
            data[_label] = _items
        else:
            if _type is not None:
                try:
                    _type = np.dtype(_type)
                except TypeError, ex:
                    _type = np.float16
            else:
                _type = np.float16
            series = _generate_distribution(column, length, _type)
            if series is not None:
                data[_label] = series
    return pd.DataFrame(data)


def generate(conf_file, output_file):
    df = generate_pandas(conf_file)
    # Override output file if it is specified in specification
    if output_file:
        output = output_file
    else:
        with open(conf_file, 'r') as stream:
            try:
                conf = yaml.safe_load(stream)
                output = _get_param(conf, 'output')
            except yaml.YAMLError as exc:
                raise ConfigurationException(exc.message, conf_file)
    print('Output to %s' % (output if output else 'stdout'))
    if output:
        df.to_csv(output, sep=";", header=True, index=False, encoding='utf-8')
    else:
        print(df)


def main(argv):
    parser = ArgumentParser(description='Generate artificial datasets which can be used for machine learning tasks')
    parser.add_argument('-o', '--output-file', required=False,
                        help='Output CSV file; overrides the one which is given in the specification')
    parser.add_argument("specification", nargs=1)
    args = parser.parse_args()
    if len(args.specification) != 1:
        print("Error: dataset specification YAML file is not specified")
    else:
        print('Generating dataset...')
        generate(args.specification[0], args.output_file)

if __name__ == "__main__":
    main(sys.argv[1:])

