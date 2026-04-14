package flash.utils {
    [Ruffle(InstanceAllocator)]
    public dynamic class Dictionary {
        prototype.toJSON = function(r:String):* {
            return "Dictionary";
        };
        prototype.setPropertyIsEnumerable("toJSON", false);

        private native function initWeakKeys():void;

        public function Dictionary(weakKeys:Boolean = false) {
            if (weakKeys) {
                this.initWeakKeys();
            }
        }
    }
}
