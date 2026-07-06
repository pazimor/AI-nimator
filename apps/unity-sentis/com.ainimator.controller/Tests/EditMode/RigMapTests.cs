using NUnit.Framework;
using UnityEngine;
using AInimator.Controller.Rig;
using AInimator.Controller.Editor;

namespace AInimator.Controller.Tests
{
    /// <summary>
    /// Tests for <see cref="RigMap"/> (22-entry structure, minimal-mapping
    /// check) and <see cref="RigMapAutoMapper"/> (best-effort name matching,
    /// <c>apps/spec/rig_binding.md</c> §2.1), using a small synthetic
    /// GameObject hierarchy built and torn down per test.
    /// </summary>
    public class RigMapTests
    {
        private GameObject _rigRoot;

        [TearDown]
        public void TearDown()
        {
            if (_rigRoot != null)
            {
                Object.DestroyImmediate(_rigRoot);
            }
        }

        [Test]
        public void NewRigMap_HasExactly22Entries_InCanonicalOrder()
        {
            var map = ScriptableObject.CreateInstance<RigMap>();
            map.EnsureEntriesSized();

            Assert.That(map.Entries.Length, Is.EqualTo(Smpl22Skeleton.NumBones));
            for (var i = 0; i < map.Entries.Length; i++)
            {
                Assert.That(map.Entries[i].smplBoneName, Is.EqualTo(RigMap.Smpl22BoneNames[i]));
            }

            Object.DestroyImmediate(map);
        }

        [Test]
        public void HasMinimalMapping_FalseUntilPelvisAssigned()
        {
            var map = ScriptableObject.CreateInstance<RigMap>();
            map.EnsureEntriesSized();

            Assert.That(map.HasMinimalMapping(), Is.False);

            var pelvisGo = new GameObject("pelvis_bone");
            map.Entries[Smpl22Skeleton.Pelvis].rigBone = pelvisGo.transform;

            Assert.That(map.HasMinimalMapping(), Is.True);

            Object.DestroyImmediate(pelvisGo);
            Object.DestroyImmediate(map);
        }

        [Test]
        public void GetRigBone_UnassignedEntry_ReturnsNull()
        {
            var map = ScriptableObject.CreateInstance<RigMap>();
            map.EnsureEntriesSized();

            Assert.That(map.GetRigBone(Smpl22Skeleton.LeftElbow), Is.Null);

            Object.DestroyImmediate(map);
        }

        [Test]
        public void GetRigBone_OutOfRangeIndex_Throws()
        {
            var map = ScriptableObject.CreateInstance<RigMap>();
            map.EnsureEntriesSized();

            Assert.Throws<System.ArgumentOutOfRangeException>(() => map.GetRigBone(-1));
            Assert.Throws<System.ArgumentOutOfRangeException>(() => map.GetRigBone(Smpl22Skeleton.NumBones));

            Object.DestroyImmediate(map);
        }

        [Test]
        public void AutoMap_MatchesMixamoStyleBoneNames()
        {
            _rigRoot = BuildMixamoStyleHierarchy();
            var map = ScriptableObject.CreateInstance<RigMap>();
            map.EnsureEntriesSized();

            var mapped = RigMapAutoMapper.AutoMap(map, _rigRoot.transform);

            Assert.That(mapped, Is.GreaterThanOrEqualTo(5));
            Assert.That(map.GetRigBone(Smpl22Skeleton.Pelvis)?.name, Is.EqualTo("Hips"));
            Assert.That(map.GetRigBone(Smpl22Skeleton.Head)?.name, Is.EqualTo("Head"));
            Assert.That(map.GetRigBone(Smpl22Skeleton.LeftShoulder)?.name, Is.EqualTo("LeftArm"));
            Assert.That(map.GetRigBone(Smpl22Skeleton.RightShoulder)?.name, Is.EqualTo("RightArm"));

            Object.DestroyImmediate(map);
        }

        [Test]
        public void AutoMap_LeavesEntryEmptyWhenNoMatchFound()
        {
            _rigRoot = new GameObject("EmptyRig");
            var map = ScriptableObject.CreateInstance<RigMap>();
            map.EnsureEntriesSized();

            var mapped = RigMapAutoMapper.AutoMap(map, _rigRoot.transform);

            Assert.That(mapped, Is.EqualTo(0));
            Assert.That(map.GetRigBone(Smpl22Skeleton.Pelvis), Is.Null);

            Object.DestroyImmediate(map);
        }

        [Test]
        public void AutoMap_DoesNotOverwriteAlreadyAssignedEntries()
        {
            _rigRoot = BuildMixamoStyleHierarchy();
            var map = ScriptableObject.CreateInstance<RigMap>();
            map.EnsureEntriesSized();

            var customPelvis = new GameObject("MyCustomPelvis");
            map.Entries[Smpl22Skeleton.Pelvis].rigBone = customPelvis.transform;

            RigMapAutoMapper.AutoMap(map, _rigRoot.transform);

            Assert.That(map.GetRigBone(Smpl22Skeleton.Pelvis)?.name, Is.EqualTo("MyCustomPelvis"));

            Object.DestroyImmediate(customPelvis);
            Object.DestroyImmediate(map);
        }

        private static GameObject BuildMixamoStyleHierarchy()
        {
            var root = new GameObject("RigRoot");
            var hips = NewChild(root.transform, "Hips");
            NewChild(hips, "Spine");
            NewChild(hips, "Head");
            NewChild(root.transform, "LeftArm");
            NewChild(root.transform, "RightArm");
            return root;
        }

        private static Transform NewChild(Transform parent, string name)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            return go.transform;
        }
    }
}
