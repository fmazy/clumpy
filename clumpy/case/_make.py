from .. import Territory
from .. import Region
from .. import Land
from ..transition_probability_estimation import Bayes
from ..density_estimation import _methods as _density_estimation_methods
from ..allocation import Unbiased, UnbiasedMonoPixel
from ..allocation import _methods as _allocation_methods
from .. import feature_selection

import inspect


def extract_parameters(func, kwargs):
    sig = inspect.signature(func)
    parameters = {}
    func_parameters = list(sig.parameters.keys())

    for key, value in kwargs.items():
        if key in func_parameters:
            parameters[key] = value

    return (parameters)


def make_default_territory(transition_matrices,
                           features=[],
                           feature_selector=None,
                           density_estimation_method='bkde',
                           allocation_method='unbiased',
                           verbose=0,
                           **kwargs):
    territory = Territory(verbose=verbose)

    for region_label, tm in transition_matrices.items():
        region = Region(label=region_label,
                        verbose=verbose,
                        verbose_heading_level=2)
        territory.add_region(region)

        # for each transitions
        for state_u in tm.palette_u:

            # if density_estimation_method == 'gkde'
            # density_estimator =

            de_class = _density_estimation_methods[density_estimation_method]
            de_parameters = extract_parameters(de_class, kwargs)

            tpe = Bayes(density_estimator=de_class(verbose=verbose,
                                                   **de_parameters),
                        verbose=1,
                        verbose_heading_level=4)

            for state_v in tm.palette_v:
                # if the transition is expected
                if tm.get(state_u, state_v) > 0.0:
                    add_cde_parameters = extract_parameters(tpe.add_conditional_density_estimator, kwargs)

                    cde_class = _density_estimation_methods[density_estimation_method]
                    cde_parameters = extract_parameters(cde_class, kwargs)

                    tpe.add_conditional_density_estimator(
                        state=state_v,
                        density_estimator=cde_class(verbose=verbose,
                                                    # verbose_heading_level=5,
                                                    **cde_parameters),
                        **add_cde_parameters)

                alloc_class = _allocation_methods[allocation_method]
                alloc_parameters = extract_parameters(alloc_class, kwargs)

                allocator = alloc_class(verbose=verbose,
                                        verbose_heading_level=3,
                                        **alloc_parameters)

                # features
                # distance to the studied state_u are removed
                features_u = features.copy()
                if state_u in features_u:
                    features_u.remove(state_u)

                land_parameters = extract_parameters(Land, kwargs)

                land = Land(features=features_u,
                            feature_selector=feature_selector,
                            transition_probability_estimator=tpe,
                            allocator=allocator,
                            verbose=verbose,
                            verbose_heading_level=4,
                            **land_parameters)

                region.add_land(state=state_u,
                                land=land)

    return territory
