//! Bridge `datacode_abi::AbiValue` (SDK / dylib) ↔ [`crate::vm_value::Value`] for native shims.

use datacode_abi::AbiValue;
use crate::plugin_abi_bridge::PluginAbiBridgeContext;
use crate::vm_value::Value;

pub fn shim_with(f: fn(&[Value]) -> Value, args: &[AbiValue]) -> AbiValue {
    let mut bridge = PluginAbiBridgeContext::new();
    let mut values = Vec::with_capacity(args.len());
    for &a in args {
        match bridge.abi_to_value(a) {
            Ok(v) => values.push(v),
            Err(_) => return AbiValue::Null,
        }
    }
    let out = f(&values);
    match bridge.value_to_abi(&out) {
        Ok(av) => av,
        Err(e) => {
            crate::native_error::set_native_error(format!("ABI return conversion failed: {:?}", e));
            AbiValue::Null
        }
    }
}
