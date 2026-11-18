# Third-party notices and licensing review

Original review: 2026-10-28. Release-copy verification: 2026-11-18. These notices supplement the individual source headers; they do not replace them, remove their conditions, or transfer upstream copyrights to G2M-LIO.

## AKF-LIO

G2M-LIO is derived from AKF-LIO by Xupeng Xie and collaborators.

- Upstream: https://github.com/xpxie/AKF-LIO
- Inspected committed revision: `45fc811ba6fe849e78485076cd0e0c15655e5791`. The baseline comparison uses that committed revision rather than uncommitted edits in the local upstream checkout.
- Its `LICENSE` contains the GNU GPL version 2. Our existing package license was already an exact copy; the root [LICENSE](LICENSE) now contains that same unmodified text.
- Its `package.xml` declares `BSD`. The committed LICENSE and package.xml were also fetched from the exact baseline revision on 2026-11-18; the GPL v2 text matches this repository LICENSE, and package.xml still declares BSD. Its README contains an academic citation request but no explicit license clarification in the prior local review. These facts do not establish a separate permissive license for all AKF-LIO source files.
- G2M-LIO elects GPL-2.0-only for its derivative program and project changes. This election is not a representation that upstream explicitly used that SPDX identifier or that the upstream ambiguity has been resolved. Seek confirmation from the upstream copyright holders before public code distribution.
- The upstream README credits LOAM, FAST-LIO2, and Faster-LIO. Those acknowledgements are recorded here as provenance, not as a claim that their entire repositories are included.
- Preserve upstream ownership, applicable license notices and warranty disclaimers. The existing upstream author entry is retained in the ROS package metadata.

The upstream README contains an academic citation request. That request is recorded as provenance, not as an added GPL condition.

## G2M-LIO changes and provenance

G2M-LIO adapts the AKF-LIO mapping pipeline and package namespace, adds map-point covariance propagation, and changes map/filter integration, preprocessing, IMU processing, configuration, and visualization. Existing file-specific upstream copyright and license notices are retained. These descriptions do not claim original authorship of inherited files.

The source release prepared on 2026-11-18 includes the current development working tree, including changes beyond development HEAD ef36140. It is arranged as a ROS package at the repository root, with bundled components, tests, and four sensor configurations. Build metadata, dependency declarations, installation instructions, public paths, and output-directory tracking were prepared for distribution; no algorithm fixes are implied by these packaging changes. Dated file-level modification notices remain in place.

## Separately licensed bundled components

| Component | Included paths | Observed license | Treatment |
| --- | --- | --- | --- |
| IKFoM / MTK | `include/IKFoM_toolkit/` | BSD-style three-condition notices | Original file-specific holders and clauses are retained, including University of Hong Kong, Universitaet Bremen, and DFKI notices. A dated G2M-LIO modification notice is added to the modified `esekfom.hpp`; the existing BSD notice remains intact. |
| Hilbert curve implementation | `include/ivox3d/hilbert.hpp` | MIT; Copyright (c) 2019 David Beynon | Existing full notice is retained. The header identifies `spectral3d/hilbert_hpp` as its source. |
| Ceres FindGlog module | `cmake/FindGlog.cmake` | BSD-style three-condition notice; Copyright 2015 Google Inc. | Existing full notice is retained. |
| Livox ROS message package | `thirdparty/livox_ros_driver/` | MIT in its package metadata | The included package, CMake file, and message definitions remain separately attributed. The original upstream LICENSE.txt is included in thirdparty/livox_ros_driver/. The two bundled message definitions match upstream apart from whitespace. Existing copyright notices are retained. |

The bundled Livox directory contains CustomMsg.msg, CustomPoint.msg, CMakeLists.txt, package.xml, and the original upstream LICENSE.txt. No Livox SDK, Boost, or FastCRC implementation is bundled here. The upstream license file also discusses components of the full upstream distribution; including that unmodified file does not imply those components are bundled in G2M-LIO.

The upstream license and message definitions were retrieved on 2026-11-18 from https://github.com/Livox-SDK/livox_ros_driver (master). The downloaded LICENSE.txt SHA-256 is a59519135db72316a2055c30723a102e9970baa2457e39078c413a8cca9d4ef1. Message definitions matched after removing whitespace. The upstream AKF-LIO metadata discrepancy remains recorded above; no additional confirmation from its copyright holders is claimed.

## Preserved notice texts

The following notices are copied from the bundled files so they can accompany distributions. Their source-file versions remain authoritative for the corresponding files.

### IKFoM / MTK notice 1

Applies to:

- `include/IKFoM_toolkit/esekfom/esekfom.hpp`
- `include/IKFoM_toolkit/esekfom/util.hpp`

```text
/*
 *  Copyright (c) 2019--2023, The University of Hong Kong
 *  All rights reserved.
 *
 *  Author: Dongjiao HE <hdj65822@connect.hku.hk>
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Universitaet Bremen nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */
```

### IKFoM / MTK notice 2

Applies to:

- `include/IKFoM_toolkit/mtk/build_manifold.hpp`
- `include/IKFoM_toolkit/mtk/src/SubManifold.hpp`
- `include/IKFoM_toolkit/mtk/src/mtkmath.hpp`
- `include/IKFoM_toolkit/mtk/startIdx.hpp`
- `include/IKFoM_toolkit/mtk/types/S2.hpp`
- `include/IKFoM_toolkit/mtk/types/SOn.hpp`
- `include/IKFoM_toolkit/mtk/types/vect.hpp`

```text
/*
 *  Copyright (c) 2019--2023, The University of Hong Kong
 *  All rights reserved.
 *
 *  Modifier: Dongjiao HE <hdj65822@connect.hku.hk>
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Universitaet Bremen nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */
```

### IKFoM / MTK notice 3

Applies to:

- `include/IKFoM_toolkit/mtk/build_manifold.hpp`
- `include/IKFoM_toolkit/mtk/src/SubManifold.hpp`
- `include/IKFoM_toolkit/mtk/src/mtkmath.hpp`
- `include/IKFoM_toolkit/mtk/src/vectview.hpp`
- `include/IKFoM_toolkit/mtk/startIdx.hpp`
- `include/IKFoM_toolkit/mtk/types/S2.hpp`
- `include/IKFoM_toolkit/mtk/types/SOn.hpp`
- `include/IKFoM_toolkit/mtk/types/vect.hpp`

```text
/*
 *  Copyright (c) 2008--2011, Universitaet Bremen
 *  All rights reserved.
 *
 *  Author: Christoph Hertzberg <chtz@informatik.uni-bremen.de>
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Universitaet Bremen nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */
```

### IKFoM / MTK notice 4

Applies to:

- `include/IKFoM_toolkit/mtk/types/wrapped_cv_mat.hpp`

```text
/*
 *  Copyright (c) 2010--2011, Universitaet Bremen and DFKI GmbH
 *  All rights reserved.
 *
 *  Author: Rene Wagner <rene.wagner@dfki.de>
 *          Christoph Hertzberg <chtz@informatik.uni-bremen.de>
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Universitaet Bremen nor the DFKI GmbH
 *     nor the names of its contributors may be used to endorse or
 *     promote products derived from this software without specific
 *     prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */
```

### Hilbert implementation

```text
// 2021-09-18, https://github.com/spectral3d/hilbert_hpp is under MIT license.

//Copyright (c) 2019 David Beynon
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:
//
//The above copyright notice and this permission notice shall be included in all
//copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.
```

### Ceres FindGlog module

```text
# Ceres Solver - A fast non-linear least squares minimizer
# Copyright 2015 Google Inc. All rights reserved.
# http://ceres-solver.org/
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of Google Inc. nor the names of its contributors may be
#   used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
```

## Distribution and remaining limits

GPL v2 permits modification subject to its terms. When distributing a derivative program, preserve required notices and distribute the covered whole under GPL v2. Binary distributions must use a valid corresponding-source arrangement under section 3; this project plans to accompany its binary releases with the complete corresponding source, including required modules, interfaces, and build/install scripts. Publishing only selected source files does not satisfy that obligation for a binary built from omitted GPL-covered modules. The license contains a system-component exception; it does not require publication of unrelated independent work merely stored in the same workspace.

Private development does not itself impose a duty to publish code. A paper-acceptance release plan cannot restrict rights in copies already distributed. Commercial use is not excluded by GPL v2. Attribution requests in a README do not replace license compliance.

The review did not establish complete compliance for all independently added scripts, linked dependencies, historical contributions, or dataset/media redistribution. The AKF-LIO GPL/BSD conflict and the missing complete Livox license notice remain unresolved upstream issues. This record documents the files inspected and the corrections made; it is not a legal clearance certificate.

License reference: https://spdx.org/licenses/GPL-2.0-only.html (sections 0-3 and 6).
