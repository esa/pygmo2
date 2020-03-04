// Copyright 2020 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <pagmo/detail/make_unique.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/s_policy.hpp>
#include <pagmo/types.hpp>

#include "common_utils.hpp"
#include "handle_thread_py_exception.hpp"
#include "object_serialization.hpp"
#include "s_policy.hpp"

namespace pagmo
{

namespace detail
{

namespace py = pybind11;

s_pol_inner<py::object>::s_pol_inner(const py::object &o)
{
    // Forbid the use of a pygmo.s_policy as a UDSP.
    // The motivation here is consistency with C++. In C++, the use of
    // a pagmo::s_policy as a UDSP is forbidden and prevented by the fact
    // that the generic constructor from UDSP is disabled if the input
    // object is a pagmo::s_policy (the copy/move constructor is
    // invoked instead). In order to achieve an equivalent behaviour
    // in pygmo, we throw an error if o is an s_policy, and instruct
    // the user to employ the standard copy/deepcopy facilities
    // for creating a copy of the input s_policy.
    if (pygmo::type(o).equal(py::module::import("pygmo").attr("s_policy"))) {
        pygmo::py_throw(PyExc_TypeError,
                        ("a pygmo.s_policy cannot be used as a UDSP for another pygmo.s_policy (if you need to copy a "
                         "selection policy please use the standard Python copy()/deepcopy() functions)"));
    }
    // Check that o is an instance of a class, and not a type.
    check_not_type(o, "s_policy");
    check_mandatory_method(o, "select", "s_policy");
    m_value = pygmo::deepcopy(o);
}

std::unique_ptr<s_pol_inner_base> s_pol_inner<py::object>::clone() const
{
    // This will make a deep copy using the ctor above.
    return detail::make_unique<s_pol_inner>(m_value);
}

individuals_group_t s_pol_inner<py::object>::select(const individuals_group_t &inds, const vector_double::size_type &nx,
                                                    const vector_double::size_type &nix,
                                                    const vector_double::size_type &nobj,
                                                    const vector_double::size_type &nec,
                                                    const vector_double::size_type &nic, const vector_double &tol) const
{
    // NOTE: select() may be called from a separate thread in pagmo::island, need to construct a GTE before
    // doing anything with the interpreter (including the throws in the checks below).
    pygmo::gil_thread_ensurer gte;

    // NOTE: every time we call into the Python interpreter from a separate thread, we need to
    // handle Python exceptions in a special way.
    std::string s_pol_name;
    try {
        s_pol_name = get_name();
    } catch (const py::error_already_set &eas) {
        pygmo::handle_thread_py_exception("Could not fetch the name of a pythonic selection policy. The error is:\n",
                                          eas);
    }

    try {
        // Fetch the new individuals in Python form.
        auto o = m_value.attr("select")(pygmo::inds_to_tuple(inds), nx, nix, nobj, nec, nic,
                                        pygmo::vector_to_ndarr<py::array_t<double>>(tol));

        // Convert back to C++ form and return.
        return pygmo::iterable_to_inds(o);
    } catch (const py::error_already_set &eas) {
        pygmo::handle_thread_py_exception(
            "The select() method of a pythonic selection policy of type '" + s_pol_name + "' raised an error:\n", eas);
    }
}

std::string s_pol_inner<py::object>::get_name() const
{
    return getter_wrapper<std::string>(m_value, "get_name", pygmo::str(pygmo::type(m_value)));
}

std::string s_pol_inner<py::object>::get_extra_info() const
{
    return getter_wrapper<std::string>(m_value, "get_extra_info", std::string{});
}

template <typename Archive>
void s_pol_inner<py::object>::save(Archive &ar, unsigned) const
{
    ar << boost::serialization::base_object<s_pol_inner_base>(*this);
    ar << pygmo::object_to_vchar(m_value);
}

template <typename Archive>
void s_pol_inner<py::object>::load(Archive &ar, unsigned)
{
    ar >> boost::serialization::base_object<s_pol_inner_base>(*this);
    std::vector<char> v;
    ar >> v;
    m_value = pygmo::vchar_to_object(v);
}

} // namespace detail

} // namespace pagmo

PAGMO_S11N_S_POLICY_IMPLEMENT(pybind11::object)

namespace pygmo
{

namespace py = pybind11;

// Serialization support for the s_policy class.
py::tuple s_policy_pickle_getstate(const pagmo::s_policy &s)
{
    // The idea here is that first we extract a char array
    // into which s has been serialized, then we turn
    // this object into a Python bytes object and return that.
    std::ostringstream oss;
    {
        boost::archive::binary_oarchive oarchive(oss);
        oarchive << s;
    }
    auto str = oss.str();
    return py::make_tuple(py::bytes(str.data(), boost::numeric_cast<py::size_t>(str.size())));
}

pagmo::s_policy s_policy_pickle_setstate(py::tuple state)
{
    // Similarly, first we extract a bytes object from the Python state,
    // and then we build a C++ string from it. The string is then used
    // to deserialized the object.
    if (py::len(state) != 1) {
        pygmo::py_throw(PyExc_ValueError, ("the state tuple passed for selection policy deserialization "
                                           "must have 1 element, but instead it has "
                                           + std::to_string(py::len(state)) + " element(s)")
                                              .c_str());
    }

    auto ptr = PyBytes_AsString(state[0].ptr());
    if (!ptr) {
        pygmo::py_throw(PyExc_TypeError, "a bytes object is needed to deserialize a selection policy");
    }

    std::istringstream iss;
    iss.str(std::string(ptr, ptr + py::len(state[0])));
    pagmo::s_policy s;
    {
        boost::archive::binary_iarchive iarchive(iss);
        iarchive >> s;
    }

    return s;
}

} // namespace pygmo
