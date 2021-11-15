from nipype.pipeline.engine import MapNode
import os
from nipype.pipeline.engine.nodes import flatten,ensure_list,Node,deepcopy,op,InterfaceResult,Bunch,socket, \
    _node_runner, str2bool,_save_resultfile,glob,shutil, logger
import hashlib

class MapNode(MapNode):
    def _get_cfmm_node_name(self,index):
        hash_id = hashlib.md5()
        for field in self.iterfield:
            if self.nested:
                fieldvals = flatten(ensure_list(getattr(self.inputs, field)))
            else:
                fieldvals = ensure_list(getattr(self.inputs, field))
            hash_id.update(str(fieldvals[index]).encode('utf-8'))
        nodename = "_%s_%s" % (self.name, hash_id.hexdigest())
        return nodename
    def _make_nodes(self, cwd=None):
        if cwd is None:
            cwd = self.output_dir()
        if self.nested:
            nitems = len(flatten(ensure_list(getattr(self.inputs, self.iterfield[0]))))
        else:
            nitems = len(ensure_list(getattr(self.inputs, self.iterfield[0])))
        for i in range(nitems):

            # AK: create unique name that isn't dependent on the order of the iterfield list
            nodename = self._get_cfmm_node_name(i)
            #nodename = "_%s%d" % (self.name, i)

            node = Node(
                deepcopy(self._interface),
                n_procs=self._n_procs,
                mem_gb=self._mem_gb,
                overwrite=self.overwrite,
                needed_outputs=self.needed_outputs,
                run_without_submitting=self.run_without_submitting,
                base_dir=op.join(cwd, "mapflow"),
                name=nodename,
            )
            node.plugin_args = self.plugin_args
            node.interface.inputs.trait_set(
                **deepcopy(self._interface.inputs.trait_get())
            )
            node.interface.resource_monitor = self._interface.resource_monitor
            for field in self.iterfield:
                if self.nested:
                    fieldvals = flatten(ensure_list(getattr(self.inputs, field)))
                else:
                    fieldvals = ensure_list(getattr(self.inputs, field))
                logger.debug("setting input %d %s %s", i, field, fieldvals[i])
                setattr(node.inputs, field, fieldvals[i])
            node.config = self.config
            yield i, node

    def _run_interface(self, execute=True, updatehash=False):
        """Run the mapnode interface

        This is primarily intended for serial execution of mapnode. A parallel
        execution requires creation of new nodes that can be spawned
        """
        self._check_iterfield()
        cwd = self.output_dir()
        if not execute:
            return self._load_results()

        # Set up mapnode folder names
        if self.nested:
            nitems = len(ensure_list(flatten(getattr(self.inputs, self.iterfield[0]))))
        else:
            nitems = len(ensure_list(getattr(self.inputs, self.iterfield[0])))

        # AK: get the hash based node names so that the result folders are not deleted
        nodenames = [self._get_cfmm_node_name(i) for i in range(nitems)]
        #nnametpl = "_%s{}" % self.name
        #nodenames = [nnametpl.format(i) for i in range(nitems)]

        # Run mapnode
        outdir = self.output_dir()
        result = InterfaceResult(
            interface=self._interface.__class__,
            runtime=Bunch(
                cwd=outdir,
                returncode=1,
                environ=dict(os.environ),
                hostname=socket.gethostname(),
            ),
            inputs=self._interface.inputs.get_traitsfree(),
        )
        try:
            result = self._collate_results(
                _node_runner(
                    self._make_nodes(cwd),
                    updatehash=updatehash,
                    stop_first=str2bool(
                        self.config["execution"]["stop_on_first_crash"]
                    ),
                )
            )
        except Exception as msg:
            result.runtime.stderr = "%s\n\n%s".format(
                getattr(result.runtime, "stderr", ""), msg
            )
            _save_resultfile(
                result,
                outdir,
                self.name,
                rebase=str2bool(self.config["execution"]["use_relative_paths"]),
            )
            raise

        # And store results
        _save_resultfile(result, cwd, self.name, rebase=False)
        # remove any node directories no longer required
        dirs2remove = []
        for path in glob(op.join(cwd, "mapflow", "*")):
            if op.isdir(path):
                if path.split(op.sep)[-1] not in nodenames:
                    dirs2remove.append(path)
        for path in dirs2remove:
            logger.debug('[MapNode] Removing folder "%s".', path)
            shutil.rmtree(path)

        return result