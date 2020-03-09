# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
#     or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example code for the nodes in the example pipeline. This code is meant
just for illustrating basic Kedro features.

Delete this when you start working on your own Kedro project.
"""

from kedro.pipeline import Pipeline, node
from sklearn.metrics import accuracy_score

from .metric import Metric
from .nodes import predict, report_accuracy
from fashion_benchmark.trainer import Trainer
from fashion_benchmark.models.raw_pix import raw_pix
from ...models.boosting import hist_boosting
from ...models.daisy import daisy
from ...models.hog import hog


def create_pipeline(**kwargs):
    return Pipeline([
        node(
                Trainer(raw_pix),
                ["train_images", "train_labels"],
                "raw_pix_model",
                tags=["training"]
            ),
        node(
            predict,
            ["raw_pix_model", "test_images"],
            "raw_pix_ypred",
            tags=["inference"]
        ),
        node(
            report_accuracy,
            ["raw_pix_ypred", "test_labels"],
            "raw_pix_report",
            tags=["eval"]
        ),
        node(
            Trainer(hog),
            ["train_images", "train_labels"],
            "hog_model",
            tags=["training"]
        ),
        node(
            predict,
            ["hog_model", "test_images"],
            "hog_ypred",
            tags=["inference"]
        ),
        node(
            report_accuracy,
            ["hog_ypred", "test_labels"],
            "hog_report",
            tags=["eval"]
        ),
        node(
            Trainer(daisy),
            ["train_images", "train_labels"],
            "daisy_model",
            tags=["training"]
        ),
        node(
            predict,
            ["daisy_model", "test_images"],
            "daisy_ypred",
            tags=["inference"]
        ),
        node(
            report_accuracy,
            ["daisy_ypred", "test_labels"],
            "daisy_report",
            tags=["eval"]
        ),
        node(
            Metric(accuracy_score),
            ["daisy_ypred", "test_labels"],
            "daisy_accuracy",
            tags=["eval"]
        ),
        node(
            Metric(accuracy_score),
            ["raw_pix_ypred", "test_labels"],
            "raw_pix_accuracy",
            tags=["eval"]
        ),
        node(
            Metric(accuracy_score),
            ["hog_ypred", "test_labels"],
            "hog_accuracy",
            tags=["eval"]
        ),
        node(
            Trainer(hist_boosting),
            ["train_images", "train_labels"],
            "boosting_model",
            tags=["training", "boosting"]
        ),
        node(
            predict,
            ["boosting_model", "test_images"],
            "boosting_ypred",
            tags=["inference", "boosting"]
        ),
        node(
            Metric(accuracy_score),
            ["boosting_ypred", "test_labels"],
            "boosting_accuracy",
            tags=["eval"]
        ),
    ])