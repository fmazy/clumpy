from .. import Territory
from .. import Region
from .. import Land
from ..transition_probability_estimation import Bayes
from ..density_estimation import GKDE
from ..allocation import Unbiased
from .. import feature_selection


def make_default_territory(transition_matrices,
                           features=[],
                           feature_selector=None,
                           n_jobs_predict=1,
                           n_jobs_neighbors=1,
                           n_fit_max=2 * 10 ** 4,
                           n_predict_max=2 * 10 ** 4,
                           P_v_min=0.0,
                           n_samples_min=1,
                           update_P_Y=False,
                           n_allocation_try=10 ** 3,
                           fit_bootstrap_patches=True,
                           verbose=0):
    territory = Territory(verbose=verbose)

    for region_label, tm in transition_matrices.items():
        region = Region(label=region_label,
                        verbose=verbose,
                        verbose_heading_level=2)
        territory.add_region(region)

        # for each transitions
        for state_u in tm.palette_u:

            tpe = Bayes(density_estimator=GKDE(n_jobs_predict=n_jobs_predict,
                                               n_jobs_neighbors=n_jobs_neighbors,
                                               n_predict_max=n_predict_max,
                                               n_fit_max=n_fit_max,
                                               verbose=verbose,
                                               verbose_heading_level=5),
                        verbose=verbose,
                        verbose_heading_level=4)

            for state_v in tm.palette_v:
                # if the transition is expected
                if tm.get(state_u, state_v) > 0.0:
                    tpe.add_conditional_density_estimator(state=state_v,
                                                          density_estimation=GKDE(n_jobs_predict=n_jobs_predict,
                                                                                  n_predict_max=n_predict_max,
                                                                                  n_fit_max=n_fit_max,
                                                                                  verbose=verbose,
                                                                                  verbose_heading_level=5),
                                                          P_v_min=P_v_min,
                                                          n_samples_min=n_samples_min)

                allocator = Unbiased(update_P_Y=update_P_Y,
                                     n_allocation_try=n_allocation_try,
                                     verbose=verbose)

                # features
                # distance to the studied state_u are removed
                features_u = features.copy()
                if state_u in features_u:
                    features_u.remove(state_u)

                land = Land(features=features_u,
                            feature_selector=feature_selector,
                            transition_probability_estimator=tpe,
                            fit_bootstrap_patches=fit_bootstrap_patches,
                            allocator=allocator,
                            verbose=verbose,
                            verbose_heading_level=4)

                region.add_land(state=state_u,
                                land=land)

    return territory
