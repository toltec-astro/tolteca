Execution and storage
=====================

Use execution records to retain how a result was produced, and a store to
organize saved results for later use. A run and a destination answer different
questions: the same execution can write several products to several stores.

A namespace gives an output scope a stable identity. Its store provides
filesystem access. A workflow run identifies an execution, while a store-run
record gives that execution its local name and output outcomes in one namespace.
Moving an output tree does not change those recorded identities.

The guide below covers creating a namespace, recording an execution, choosing
the appropriate write or publication operation, and reading saved run records.
It builds on the :doc:`data-product-record-model`.

.. toctree::
   :maxdepth: 1

   data-product-runs

Continue with :doc:`data-product-examples` to publish artifacts, repeat a result
and reopen either its selected or an earlier version.
